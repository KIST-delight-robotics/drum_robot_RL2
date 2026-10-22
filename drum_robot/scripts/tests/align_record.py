"""
[확인용 임시 모듈] play_align.py, play_song.py 공통: 매 스텝 env 0번의 허리·어깨 yaw 기록, CSV 저장, 요약

기록 (θ0 허리, θ1 오른 어깨 yaw, θ2 왼 어깨 yaw)
    desired    : 정렬 보상의 목표 (허리 = 쌍 표, 어깨 yaw = 같은 식으로 다시 계산: θ1* = atan2(y_R - l·sin q0, x_R - l·cos q0) - q0, 왼쪽은 부호 반대)
    RL_desired : 정책이 명령한 관절 목표 env.target_q (행동 = 목표 속도를 누적, 실제 ± max_lead 안으로 자름)
    output     : 실제 관절 각도
    + 손별 다음 음표(드럼, 남은 시간), 이번 스텝 가상 타격
환경의 _get_rewards 뒤에 붙여서 기록 (환경 코드는 그대로)
"""

import math

COLS = ["t", "waist_tgt", "waist_act", "yaw_r_tgt", "yaw_r_act", "yaw_l_tgt", "yaw_l_act",
        "note_l", "note_l_s", "note_r", "note_r_s", "hit_l", "hit_r",
        "waist_cmd", "yaw_r_cmd", "yaw_l_cmd"]


class AlignRecorder:
    def __init__(self, env, print_s: float = 0.25, on_episode_end=None, on_step=None, tag: str = "play_align"):
        # on_episode_end(rows): 리셋을 만나면 직전 에피소드 기록을 넘김
        # on_step(step, rows) : 매 스텝 기록 뒤 호출 (주기 저장, 곡 끝 종료 등)
        from drum_robot.tasks.drumrobot.components.specs import Instruments

        self.env = env
        self.al = env.reward_computer.alignment
        self.names = list(Instruments().all.keys())
        self.dt = env.step_dt
        self.horizon_s = (env.rds.env.max_lookahead_step - 1) * self.dt    # 남은 시간 1.0 = 이 초
        self.print_every = max(1, round(print_s / self.dt))
        self.on_episode_end = on_episode_end
        self.on_step = on_step
        self.tag = tag
        self.rows = []
        self.prev_step = -1

        orig = env._get_rewards
        def _get_rewards():
            reward = orig()
            self.record()
            return reward
        env._get_rewards = _get_rewards

    @staticmethod
    def _near(target, actual):
        # 목표를 실제 각도에 가장 가까운 같은 각도로 (±360° 차이 없애기, 보상의 _wrap과 같은 기준)
        d = math.remainder(target - actual, 2 * math.pi)
        return actual + d

    def record(self):
        env, al = self.env, self.al
        step = int(env.steps[0].item())
        if step <= self.prev_step:                      # 리셋됨 -> 직전 에피소드 넘김
            rows, self.rows = self.rows, []
            if self.on_episode_end and len(rows) >= 2:
                self.on_episode_end(rows)
        self.prev_step = step

        q = env.robot_interface.get_joint_pos(env.robot)[0]
        q0, q1, q2 = q[al.j_waist].item(), q[al.j_r].item(), q[al.j_l].item()
        pR = env.drum_pos[0, al.target_r[0], :2].tolist()
        pL = env.drum_pos[0, al.target_l[0], :2].tolist()
        th1 = math.atan2(pR[1] - al.l * math.sin(q0), pR[0] - al.l * math.cos(q0)) - q0
        th2 = math.atan2(pL[1] + al.l * math.sin(q0), pL[0] + al.l * math.cos(q0)) - q0
        waist_tgt = self._near(al.target_waist[0].item(), q0)
        cmd = env.target_q[0]                                                # 정책이 명령한 관절 목표 (RL)
        c0, c1, c2 = cmd[al.j_waist].item(), cmd[al.j_r].item(), cmd[al.j_l].item()

        valid = env.arm_valid[0].tolist()
        drum = env.arm_drum[0].tolist()
        t_s = (env.arm_t[0] * self.horizon_s).tolist()
        hit = env._virtual_hits()[0].any(dim=-1).tolist()                 # [왼손, 오른손] 이번 스텝 가상 타격

        row = [step * self.dt, waist_tgt, q0, self._near(th1, q1), q1, self._near(th2, q2), q2,
               drum[0] if valid[0] else -1, t_s[0] if valid[0] else float("nan"),
               drum[1] if valid[1] else -1, t_s[1] if valid[1] else float("nan"),
               int(hit[0]), int(hit[1]),
               c0, c1, c2]
        self.rows.append(row)

        if step % self.print_every == 0:
            d = math.degrees
            def note(i):
                return f"{self.names[row[i]]} {row[i + 1]:.2f}s" if row[i] >= 0 else "(없음)"
            print(f"{row[0]:5.2f}s | 다음: 왼 {note(7)}, 오른 {note(9)} | 목표/명령/실제 "
                  f"허리 {d(row[1]):+6.1f}/{d(row[13]):+6.1f}/{d(row[2]):+6.1f}° ({d(abs(row[1] - row[2])):4.1f}°) | "
                  f"오른어깨 {d(row[3]):6.1f}/{d(row[14]):6.1f}/{d(row[4]):6.1f}° ({d(abs(row[3] - row[4])):4.1f}°) | "
                  f"왼어깨 {d(row[5]):6.1f}/{d(row[15]):6.1f}/{d(row[6]):6.1f}° ({d(abs(row[5] - row[6])):4.1f}°)")

        if self.on_step:
            self.on_step(step, self.rows)

    def save_csv(self, path, rows):
        # CSV (각도 deg, 시간 s)
        with open(path, "w") as f:
            f.write(",".join(COLS) + "\n")
            for r in rows:
                vals = [r[0]] + [math.degrees(x) for x in r[1:7]] + [self.names[r[7]] if r[7] >= 0 else "", r[8],
                                                                    self.names[r[9]] if r[9] >= 0 else "", r[10], r[11], r[12]] \
                       + [math.degrees(x) for x in r[13:16]]
                f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals) + "\n")

    def summary(self, rows, label: str):
        # 평균 |오차| 전체, 치기 직전(가장 가까운 음표까지 0.1초 이하)
        def err(i):
            return [math.degrees(abs(r[i] - r[i + 1])) for r in rows]
        near = [min([x for x in (r[8], r[10]) if not math.isnan(x)], default=99.0) <= 0.1 for r in rows]
        def mean(v, mask=None):
            v = v if mask is None else [x for x, m in zip(v, mask) if m]
            return sum(v) / len(v) if v else float("nan")
        e_w, e_r, e_l = err(1), err(3), err(5)
        print(f"\n[{self.tag}] {label}  평균 |오차|            허리 {mean(e_w):5.1f}°  오른어깨 {mean(e_r):5.1f}°  왼어깨 {mean(e_l):5.1f}°")
        print(f"[{self.tag}] {' ' * len(label)}  치기 직전(0.1초 안)    허리 {mean(e_w, near):5.1f}°  오른어깨 {mean(e_r, near):5.1f}°  왼어깨 {mean(e_l, near):5.1f}°")
