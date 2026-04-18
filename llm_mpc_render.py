from dotenv import load_dotenv, find_dotenv
from typing import List, Any, Optional
import numpy as np
import os, time, ast, re, argparse, math
import pandas as pd
from scipy.spatial.transform import Rotation as R

# LangChain — uses local embeddings, no OpenAI key needed
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain.indexes import VectorstoreIndexCreator

# F1TENTH gym — upstream f110_gym registers f110-v0 with Gymnasium, not legacy gym
import gymnasium as gym
import f110_gym  # registers f110-v0 in Gymnasium's registry

# Absolute path to bundled maps — avoids FileNotFoundError regardless of cwd
_MAP_DIR = os.path.join(os.path.dirname(f110_gym.__file__), 'envs', 'maps')

load_dotenv(find_dotenv())

MODEL_OPTIONS = ['gpt-4o', 'custom', 'training']

# Shared local embedding model — loaded once, reused for both RAG indices
_LOCAL_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


class RaceLLMMPC():
    def __init__(self,
                 openai_token: str,
                 model: str,
                 model_dir: str = None,
                 quant: bool = False,
                 map_name: str = 'vegas',
                 map_ext: str = '.png',
                 # kept for API compatibility with training script — ignored
                 ros: Any = None,
                 no_ROS: bool = True,
                 use_f1tenth: bool = True,
                 f1tenth_map: str = 'vegas',
                 host_ip: str = '192.168.192.105',
                 render_sim: bool = False,
                 render_mode: str = 'human',
                 sim_rollout_steps: int = 1):
        """
        render_sim: if True, call F110Env.render() after reset/step (needs display + working pyglet/OpenGL).
        render_mode: 'human' (real-time-ish) or 'human_fast' (see f110_gym F110Env.render).
        sim_rollout_steps: after each LLM MPC update, run this many physics steps with those params.
            Default 1 is ~0.01s sim time (invisible). Use 100–300 with --render to see motion per prompt.
        """

        # ── MPC params ────────────────────────────────────────────────────────
        self.DEFAULT_MPC_PARAMS = {
            "qv": 10.0, "qn": 20.0, "qalpha": 7.0, "qac": 0.01,
            "qddelta": 0.1, "alat_max": 10.0, "a_min": -10.0,
            "a_max": 10.0, "v_min": 1.0, "v_max": 12.0,
            "track_safety_margin": 0.3
        }
        self.current_params = self.DEFAULT_MPC_PARAMS.copy()

        # ── F1TENTH gym ───────────────────────────────────────────────────────
        _map_name = f1tenth_map if use_f1tenth else map_name
        _map_abs  = os.path.join(_MAP_DIR, _map_name)  # absolute path — avoids FileNotFoundError
        self.env = gym.make(
            'f110-v0',
            map=_map_abs,
            map_ext=map_ext,
            num_agents=1,
            disable_env_checker=True,
        )
        self.obs         = None
        self.f1tenth_done = False
        self.param_history: List[dict] = []   # for Rstability reward term
        self._render_sim = render_sim
        self._render_mode = render_mode if render_mode in ('human', 'human_fast') else 'human'
        self._render_warned = False
        self._sim_rollout_steps = max(1, int(sim_rollout_steps))

        # Build a raceline from the map for Frenet calculations
        self.raceline  = self._build_raceline(_map_name)
        self.odom_hz   = 50.0   # gym equivalent; used by legacy callers

        # ROS is not used — kept as None for API compatibility
        self.ros = None

        # Reset to get initial observation (Gymnasium: options dict, returns obs, info)
        self.obs, _ = self.env.reset(
            options={'poses': np.array([[0.0, 0.0, np.pi / 2]])}
        )
        self._maybe_render()

        # ── LLM & RAG ─────────────────────────────────────────────────────────
        self.openai_token = openai_token
        self.quant        = quant
        self.llm, self.custom, self.use_openai = self.init_llm(
            model=model, model_dir=model_dir, openai_token=openai_token
        )
        self.base_memory, self.vector_index = self.load_memory()
        self.decision_index                 = self.load_decision_mem()

    # ─── LLM init ─────────────────────────────────────────────────────────────

    def init_llm(self, model: str, model_dir: str, openai_token: str) -> tuple:
        use_openai = False
        custom     = False
        llm        = None

        if model not in MODEL_OPTIONS:
            raise ValueError(f"Model {model} not supported. Use one of {MODEL_OPTIONS}")

        if model == 'gpt-4o':
            use_openai = True
            llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_token)
        elif model == 'custom':
            if self.quant:
                from inference.inf_gguf import RaceLLMGGGUF
                gguf_name = [f for f in os.listdir(model_dir) if f.endswith('.gguf')][0]
                llm = RaceLLMGGGUF(model_dir=model_dir, gguf_name=gguf_name, max_tokens=256)
            else:
                from inference.inf_pipeline import RaceLLMPipeline
                llm = RaceLLMPipeline(model_dir=model_dir, chat_template='qwen-2.5', load_in_4bit=True)
            custom = True
        elif model == 'training':
            print("[INFO] model='training': skipping LLM init — "
                  "using RaceLLMMPC only as environment/utility vessel.")
        return llm, custom, use_openai

    # ─── RAG loading (local embeddings — no OpenAI key needed) ────────────────

    def load_memory(self) -> tuple:
        base_mem_path = os.path.join('./', 'prompts/mpc_base_memory.txt')
        with open(base_mem_path, 'r') as f:
            base_mem = f.read()

        loader   = TextLoader('prompts/mpc_memory.txt')
        splitter = CharacterTextSplitter(separator='#', chunk_size=100, chunk_overlap=20)
        index    = VectorstoreIndexCreator(
            embedding=_LOCAL_EMBEDDINGS,
            text_splitter=splitter,
        ).from_loaders([loader])
        return base_mem, index

    def _maybe_render(self) -> None:
        """Open the pyglet window and draw the current state (no-op if render_sim is False)."""
        if not self._render_sim:
            return
        try:
            self.env.unwrapped.render(mode=self._render_mode)
        except Exception as e:
            if not self._render_warned:
                print(f"[WARN] Simulation rendering failed (headless, GL, or pyglet): {e}")
                self._render_warned = True

    def _rollout_with_mpc(self, merged_params: dict) -> None:
        """
        Step the sim repeatedly with the same MPC command so steering/speed actually move the car.
        A single step is one physics dt (~0.01s); without a rollout the render barely changes per prompt.
        """
        for i in range(self._sim_rollout_steps):
            if self.f1tenth_done:
                break
            self._step_f1tenth(merged_params, append_history=(i == 0))

    def load_decision_mem(self) -> VectorstoreIndexCreator:
        loader   = TextLoader('prompts/RAG_memory.txt')
        splitter = CharacterTextSplitter(separator='#', chunk_size=100, chunk_overlap=20)
        return VectorstoreIndexCreator(
            embedding=_LOCAL_EMBEDDINGS,
            text_splitter=splitter,
        ).from_loaders([loader])

    # ─── F1TENTH gym interface (used by rl_mpc_train.py) ─────────────────────

    def reset_f1tenth(self) -> dict:
        """
        Reset the sim between training rollouts.
        Equivalent to _reset_car() + _reset_mpc_params() in the ROS version.
        """
        self.obs, _ = self.env.reset(
            options={'poses': np.array([[0.0, 0.0, np.pi / 2]])}
        )
        self.f1tenth_done = False
        self.param_history = []
        self.current_params = self.DEFAULT_MPC_PARAMS.copy()
        self._maybe_render()
        return self.obs

    def _step_f1tenth(self, mpc_params: dict, append_history: bool = True) -> tuple:
        """
        Convert MPC params -> gym action, step the sim, return (obs, done).
        Equivalent to _set_ros_param() + waiting for the MPC to act in ROS.
        """
        if append_history:
            self.param_history.append(mpc_params.copy())
        self.current_params.update(mpc_params)

        steer = self._mpc_params_to_steering(mpc_params)
        speed = float(np.clip(
            mpc_params.get('v_max', 12.0) * 0.4,
            mpc_params.get('v_min', 1.0),
            mpc_params.get('v_max', 12.0),
        ))
        action = np.array([[steer, speed]])

        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.f1tenth_done = bool(terminated or truncated)
        self._maybe_render()
        return self.obs, self.f1tenth_done

    def _get_f1tenth_odom(self) -> dict:
        """
        Return an odom dict in the same format as _filter_odom() in the ROS version
        so that _dist_to_boundaries() and reward functions work unchanged.
        """
        if self.obs is None:
            return {'s_pos': [0.0], 'd_pos': [0.0],
                    's_speed': [0.0], 'd_speed': [0.0], 'theta': [0.0]}

        x     = float(self.obs['poses_x'][0])
        y     = float(self.obs['poses_y'][0])
        theta = float(self.obs['poses_theta'][0])
        vx    = float(self.obs['linear_vels_x'][0])
        vy    = float(self.obs['linear_vels_y'][0])

        s, d, _ = self._get_frenet_state(x, y)

        return {
            's_pos':   [round(s, 3)],
            'd_pos':   [round(d, 3)],
            's_speed': [round(vx, 3)],
            'd_speed': [round(vy, 3)],
            'theta':   [round(theta, 3)],
        }

    def _mpc_params_to_steering(self, params: dict) -> float:
        """
        Heuristic: lateral deviation weight (qn) vs heading weight (qalpha)
        drives steering. Output clamped to F1TENTH steering limits.
        """
        qn     = params.get('qn', 20.0)
        qalpha = params.get('qalpha', 7.0)
        raw    = (qn - qalpha) / (qn + qalpha + 1e-6)
        return float(np.clip(raw * 0.4189, -0.4189, 0.4189))

    # ─── Frenet helpers ───────────────────────────────────────────────────────

    def _build_raceline(self, map_name: str) -> dict:
        """
        Build a raceline dict matching the ROS version's format.
        Tries to load a waypoint CSV; falls back to a unit circle.
        """
        import os
        wp_candidates = [
            f'f1tenth_gym/gym/f110_gym/envs/maps/{map_name}_waypoints.csv',
            f'f1tenth_gym/maps/{map_name}_waypoints.csv',
        ]
        for wp_path in wp_candidates:
            if os.path.exists(wp_path):
                try:
                    wps    = np.loadtxt(wp_path, delimiter=',')
                    xs, ys = wps[:, 0], wps[:, 1]
                    break
                except Exception:
                    pass
        else:
            # fallback: circle track of radius 3 m
            t      = np.linspace(0, 2 * np.pi, 200, endpoint=False)
            xs, ys = 3.0 * np.cos(t), 3.0 * np.sin(t)

        diffs = np.diff(np.stack([xs, ys], axis=1), axis=0)
        ds    = np.linalg.norm(diffs, axis=1)
        ss    = np.concatenate([[0.0], np.cumsum(ds)])
        psis  = np.arctan2(np.gradient(ys), np.gradient(xs))

        # approximate boundary widths (F1TENTH car width ~0.31 m, track ~1.6 m)
        d_lefts  = [0.8] * len(xs)
        d_rights = [0.8] * len(xs)

        return {
            's':       ss.tolist(),
            'x':       xs.tolist(),
            'y':       ys.tolist(),
            'psi':     psis.tolist(),
            'd_left':  d_lefts,
            'd_right': d_rights,
        }

    def _get_frenet_state(self, x: float, y: float) -> tuple:
        """Return (s, d, nearest_index) for a Cartesian position."""
        if not self.raceline:
            return 0.0, 0.0, 0
        xs_r = np.array(self.raceline['x'])
        ys_r = np.array(self.raceline['y'])
        dists = np.sqrt((xs_r - x) ** 2 + (ys_r - y) ** 2)
        idx   = int(np.argmin(dists))
        s     = self.raceline['s'][idx]
        psi_r = self.raceline['psi'][idx]
        dx, dy = x - xs_r[idx], y - ys_r[idx]
        d = dy * np.cos(psi_r) - dx * np.sin(psi_r)
        return s, d, idx

    def _dist_to_boundaries(self, data: dict, raceline: dict) -> tuple:
        """
        Identical interface to the ROS version — used by reward functions.
        Returns (d_left, d_right) lists.
        """
        if not raceline:
            return [1.0] * len(data.get('s_pos', [1])), \
                   [1.0] * len(data.get('s_pos', [1]))

        s_raceline = np.array(raceline['s'])
        d_left, d_right = [], []
        for s_pos, d_pos in zip(data['s_pos'], data['d_pos']):
            idx   = int(np.argmin(np.abs(s_raceline - s_pos)))
            d_l   = round(abs(raceline['d_left'][idx]  - d_pos), 3)
            d_r   = round(abs(raceline['d_right'][idx] + d_pos), 3)
            d_left.append(d_l)
            d_right.append(d_r)
        return d_left, d_right

    # ─── Methods expected by rl_mpc_train.py (stubs for gym mode) ────────────

    def _reset_mpc_params(self):
        """Reset params to defaults. No-op in gym mode (reset_f1tenth handles it)."""
        self.current_params = self.DEFAULT_MPC_PARAMS.copy()

    def _reset_car(self):
        """Alias for reset_f1tenth — called by reward function on crash."""
        self.reset_f1tenth()

    def _set_ros_param(self, param: str, value: float, suppress_print: bool = False):
        """In gym mode: update current_params dict only (no ROS call)."""
        if param in self.DEFAULT_MPC_PARAMS:
            self.current_params[param] = value

    def _crash_detection_via_sim(self) -> bool:
        """Return gym collision flag."""
        return self.f1tenth_done

    def _mpc_crash_detection(self, echo_nb: int = 200, timeout: float = 0.5) -> bool:
        """In gym mode: no MPC process to crash."""
        return False

    # ─── Core LLM methods ────────────────────────────────────────────────────

    def race_mpc_interact(self, scenario: str, memory_nb: int = 0,
                          prompt_only: bool = False) -> tuple:
        """
        Tunes MPC parameters based on the scenario using the LLM.
        Same interface as the original ROS version.
        """
        RAG_query  = f"Task: {scenario}"
        rag_sources = []
        if memory_nb > 0 and self.vector_index:
            docs = self.vector_index.vectorstore.search(
                query=RAG_query, search_type='similarity', k=memory_nb)
            rag_sources = [{'meta': d.metadata, 'content': d.page_content} for d in docs]

        LLM_query = f"""
You are an AI assistant tuning MPC parameters for an autonomous racing car.

## Context
1. Scenario: {scenario}
2. Base Memory: {self.base_memory}

## RAG Hints
{rag_sources}

## Task
Adapt the tuneable MPC parameters. Return ONLY the dictionary below (no comments):
new_mpc_params = {{
    'param1': new_value1,
    'param2': new_value2,
}}
"""
        if prompt_only:
            return None, None, rag_sources, LLM_query, None

        if self.custom:
            llm_out, _, _ = self.llm(LLM_query)
        else:
            response = self.llm.invoke(LLM_query)
            llm_out  = response.content if hasattr(response, 'content') else response

        extracted_command, llm_expl = self._sanitize_tune_output(llm_out)

        if extracted_command:
            for p, v in extracted_command.items():
                self._set_ros_param(p, v)

            merged = {**self.DEFAULT_MPC_PARAMS, **extracted_command}
            # Multiple steps so the window reflects the new behavior (one step ≈ imperceptible motion).
            self._rollout_with_mpc(merged)

        return extracted_command, llm_expl, rag_sources, LLM_query, llm_out

    def race_reasoning(self, human_prompt: str, data_time: float = 2.0,
                       data_samples: int = 5, prompt_only: bool = False,
                       k: int = 5) -> str:
        """
        Decision-making: reads current gym state, queries LLM for action.
        Same return interface as original ROS version.
        """
        odom_data = self._get_f1tenth_odom()
        # replicate over data_samples so downstream code gets a list
        odom_data = {key: val * data_samples for key, val in odom_data.items()}

        d_left, d_right = self._dist_to_boundaries(odom_data, self.raceline)
        reversing_bool  = float(np.mean(odom_data['s_speed'])) < -0.1
        crashed_bool    = self.f1tenth_done
        facing_wall     = False   # gym provides collision via done flag

        rag_sources = []
        if k > 0 and self.decision_index:
            docs        = self.decision_index.vectorstore.search(
                query=human_prompt, search_type='similarity', k=k)
            rag_sources = docs
        hints = "".join([doc.page_content + "\n" for doc in rag_sources])

        prompt = f"""
You are an AI pilot embodied on an autonomous racing car.
Human command: {human_prompt}

Car data (Frenet frame):
- s_pos (progress): {odom_data['s_pos']}
- d_pos (lat error): {odom_data['d_pos']}
- s_speed: {odom_data['s_speed']}
- d_speed: {odom_data['d_speed']}
- Distance to left wall: {d_left}
- Distance to right wall: {d_right}
- Reversing: {reversing_bool}
- Crashed: {crashed_bool}

Hints: {hints}

Choose one action:
- "Continue behavior"
- "Change behavior: <brief instruction>"

Output: "Action": <your choice>
"""
        if prompt_only:
            return prompt

        if self.custom:
            llm_out, _, _ = self.llm(prompt)
        else:
            response = self.llm.invoke(prompt)
            llm_out  = response.content if hasattr(response, 'content') else response

        state_str = (f"d_pos: {odom_data['d_pos'][-1]}, "
                     f"s_speed: {odom_data['s_speed'][-1]}, "
                     f"d_left: {d_left[-1]}, d_right: {d_right[-1]}, "
                     f"crashed: {crashed_bool}")
        return llm_out, state_str

    # ─── Output parsing (unchanged from original) ──────────────────────────────

    def _sanitize_tune_output(self, text: str) -> tuple:
        command_dict     = None
        explanation_text = text
        try:
            dict_start_idxs = [m.start() for m in re.finditer(r'new_mpc_params\s*=\s*{', text)]
            dict_end_idxs   = [m.end()   for m in re.finditer(r'}', text)]
            for start_idx in dict_start_idxs:
                for end_idx in dict_end_idxs:
                    if end_idx > start_idx:
                        dict_text = text[start_idx:end_idx]
                        try:
                            dict_str     = dict_text.split('=', 1)[1].strip()
                            command_dict = ast.literal_eval(dict_str)
                            if isinstance(command_dict, dict) and len(command_dict) > 0:
                                return command_dict, explanation_text
                        except (ValueError, SyntaxError):
                            continue
        except Exception as e:
            print(f"Parse error: {text}, {e}")
        return command_dict, explanation_text


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    OPENAI_API_TOKEN = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',      choices=MODEL_OPTIONS, default='custom')
    parser.add_argument('--model_dir',  type=str, default="nibauman/RobotxLLM_Qwen7B_SFT")
    parser.add_argument('--quant',      action='store_true')
    parser.add_argument('--prompt',     type=str, default='Drive forwards at 2 m/s')
    parser.add_argument('--map_name',   type=str, default='vegas')
    parser.add_argument('--mpconly',    action='store_true')
    parser.add_argument(
        '--render',
        action='store_true',
        help='Show F1TENTH pyglet window after each reset/step (needs local display; pyglet/OpenGL).',
    )
    parser.add_argument(
        '--render_mode',
        choices=('human', 'human_fast'),
        default='human',
        help="F110Env.render mode: 'human' syncs roughly to real time; 'human_fast' draws as fast as possible.",
    )
    parser.add_argument(
        '--sim_rollout_steps',
        type=int,
        default=None,
        metavar='N',
        help='Physics steps after each successful MPC update. With --render, default is 150; otherwise 1. '
             'Higher N = more visible motion per prompt (~0.01s sim time per step).',
    )
    args = parser.parse_args()
    rollout_steps = args.sim_rollout_steps
    if rollout_steps is None:
        rollout_steps = 150 if args.render else 1
    if args.render:
        print(f"[info] Rendering: {rollout_steps} sim steps per successful MPC update "
              f"(~{rollout_steps * 0.01:.1f}s sim time; tune with --sim_rollout_steps)")

    race_llm = RaceLLMMPC(
        openai_token=OPENAI_API_TOKEN,
        model=args.model,
        model_dir=args.model_dir,
        quant=args.quant,
        f1tenth_map=args.map_name,
        render_sim=args.render,
        render_mode=args.render_mode,
        sim_rollout_steps=rollout_steps,
    )

    if args.mpconly:
        while True:
            prompt = input("Enter prompt: ")
            t0 = time.time()
            extracted, _, _, _, _ = race_llm.race_mpc_interact(scenario=prompt, memory_nb=2)
            print(f"MPCxLLM commands: {extracted}")
            print(f"Time: {time.time() - t0:.2f}s\n")
    else:
        while True:
            t0 = time.time()
            action, state = race_llm.race_reasoning(
                human_prompt=args.prompt, data_time=1.0, data_samples=5)
            print(f"State:  {state}")
            print(f"Action: {action}")
            if "Continue" not in action:
                race_llm.race_mpc_interact(scenario=action, memory_nb=2)
            time.sleep(0.05)