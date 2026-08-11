# python3 drumrobot_client/main.py
import socket

HOST = '127.0.0.1'
PORT = 1951

# 관절 정의: 인덱스 순서 = GET_STATUS 응답 순서 = motors.json id 순서
# (name, min_deg, max_deg)
JOINTS = [
    ("waist",            -90.0,  90.0),
    ("right_shoulder_1",   0.0, 150.0),
    ("left_shoulder_1",   30.0, 180.0),
    ("right_shoulder_2", -60.0,  90.0),
    ("right_elbow",        0.0, 140.1),
    ("left_shoulder_2",  -60.0,  90.0),
    ("left_elbow",         0.0, 140.1),
    ("right_wrist",      -90.0, 100.0),
    ("left_wrist",       -90.0, 100.0),
    ("right_pedal",      -90.0, 200.0),
    ("left_pedal",       -90.0, 200.0),
    ("head_yaw",         -90.0,  90.0),
    ("head_pitch",      -100.0,  90.0),
]

DEFAULT_MOVE_TIME = 3.0

def send(s, packet):
    s.sendall((packet + "\n").encode())

def get_status(s):
    """GET_STATUS 전송 후 (state, [angle_deg,...], speed, pause) 반환. 실패 시 (None, [], None, None).

    서버 응답 형식: STATUS|<state>|<q0>|...|<q12>|<speed>|<pause_valid>|<pause_id>|<pause_bar>
    pause는 재개 지점이 있으면 {"id": 곡id, "bar": 마디} 없으면 None.
    (구버전 서버는 speed까지만 보낼 수 있어 필드 수로 판별한다.)
    """
    send(s, "GET_STATUS")
    data = s.recv(4096).decode().strip()
    parts = data.split("|")
    if not parts or parts[0] != "STATUS":
        print(f"  예기치 않은 응답: {data}")
        return None, [], None, None
    state = parts[1] if len(parts) > 1 else "UNKNOWN"

    fields = parts[2:]
    speed = None
    pause = None
    if len(fields) == len(JOINTS) + 4:
        # 신형: 관절각 + speed + pause_valid + pause_id + pause_bar
        if fields[-3] == "1":
            try:
                pause = {"id": fields[-2], "bar": int(fields[-1])}
            except ValueError:
                pause = None
        try:
            speed = float(fields[-4])
        except ValueError:
            speed = None
        fields = fields[:-4]
    elif len(fields) == len(JOINTS) + 1:
        # 구형: 관절각 + speed
        try:
            speed = float(fields[-1])
        except ValueError:
            speed = None
        fields = fields[:-1]

    angles = []
    for a in fields:
        try:
            angles.append(float(a))
        except ValueError:
            angles.append(None)
    return state, angles, speed, pause

def print_status(state, angles, speed=None, pause=None):
    print(f"  로봇 상태: {state}")
    if speed is not None:
        print(f"  연주 속도 배율: {speed:.2f}x")
    if pause is not None:
        print(f"  재개 지점: 곡 {pause['id']}, 마디 {pause['bar']} (RESUME 가능)")
    print("  현재 목표 관절각 (deg):")
    for i, (name, _, _) in enumerate(JOINTS):
        cur = angles[i] if i < len(angles) and angles[i] is not None else None
        cur_str = f"{cur:8.2f}" if cur is not None else "    N/A"
        print(f"    [{i:2d}] {name:18s} = {cur_str}")

def print_pending(pending):
    if not pending:
        print("  (대기 중인 변경 없음)")
        return
    print("  대기 중인 목표:")
    for idx in sorted(pending):
        name = JOINTS[idx][0]
        print(f"    [{idx:2d}] {name:18s} -> {pending[idx]:.2f} deg")

def test_mode(s):
    """관절 번호를 하나씩 입력받아 목표각을 누적하고, run 입력 시 MOVE로 일괄 전송."""
    # TODO: 팁 목표 위치를 받아 전송하기
    state, angles, _, _ = get_status(s)
    if state is None:
        return
    print("\n--- 테스트 모드 ---")
    print_status(state, angles)

    if state != "IDLE":
        print(f"  주의: 서버는 IDLE 상태에서만 MOVE를 수행합니다 (현재 {state}).")

    print("\n관절 번호(0~12)를 입력해 목표각을 설정하세요.")
    print("  명령: 번호=목표각 설정 / t=이동시간 / list=현재 목록 / run=전송 / q=취소")

    pending = {}                 # idx -> target_deg
    move_time = DEFAULT_MOVE_TIME

    while True:
        cmd = input("입력 (번호 / t / list / run / q) > ").strip().lower()

        if cmd in ("q", "quit", "cancel"):
            print("  취소했습니다.")
            return

        if cmd == "list":
            print_pending(pending)
            print(f"  이동 시간: {move_time:.2f} s")
            continue

        if cmd == "t":
            raw = input(f"  이동 시간(초) (현재 {move_time}) > ").strip()
            try:
                move_time = float(raw)
            except ValueError:
                print("    숫자를 입력하세요. 변경하지 않습니다.")
            continue

        if cmd == "run":
            if not pending:
                print("  설정된 목표가 없습니다.")
                continue
            # MOVE|name|angle|name|angle|...|move_time
            fields = ["MOVE"]
            for idx in sorted(pending):
                fields.append(JOINTS[idx][0])
                fields.append(f"{pending[idx]:.4f}")
            fields.append(f"{move_time:.4f}")
            packet = "|".join(fields)

            print("\n  전송 미리보기:")
            print_pending(pending)
            print(f"    이동 시간: {move_time:.2f} s")
            if input("  전송할까요? (y/N) > ").strip().lower() != "y":
                print("  전송 보류. 계속 편집할 수 있습니다.")
                continue
            send(s, packet)
            print("  전송 완료.")
            return

        # 관절 번호 입력
        if not cmd.isdigit():
            print("  올바른 관절 번호 또는 명령을 입력하세요.")
            continue
        idx = int(cmd)
        if idx < 0 or idx >= len(JOINTS):
            print(f"  관절 번호는 0~{len(JOINTS)-1} 범위입니다.")
            continue

        name, lo, hi = JOINTS[idx]
        cur = angles[idx] if idx < len(angles) and angles[idx] is not None else None
        cur_str = f"{cur:.2f}" if cur is not None else "N/A"
        prev = pending.get(idx)
        prev_str = f", 설정됨 {prev:.2f}" if prev is not None else ""
        raw = input(f"  [{idx}] {name} 목표각 (현재 {cur_str}{prev_str}, 범위 {lo}~{hi}, deg, 엔터=취소) > ").strip()
        if raw == "":
            continue
        try:
            val = float(raw)
        except ValueError:
            print("    숫자를 입력하세요.")
            continue
        if val < lo or val > hi:
            print(f"    범위를 벗어났습니다 ({lo}~{hi}).")
            continue
        pending[idx] = val
        print(f"    설정: {name} -> {val:.2f} deg")

def play_ctrl_mode(s):
    """연주 중(PLAYING) 제어. stop / speed 를 PLAY_CTRL 로 전송.

    서버는 PLAYING 상태에서만 PLAY_CTRL 을 수락하며, 속도 배율을 [0.5, 2.0] 로
    제한한다. 명령 전송 후 GET_STATUS 로 서버가 적용한 실제 배율을 다시 읽어
    표시하므로, 범위 밖 요청이 제한돼도 화면에 정확히 반영된다.
    """
    state, _, speed, _ = get_status(s)
    print("\n--- 연주 제어 모드 ---")
    print(f"  로봇 상태: {state}")
    if speed is not None:
        print(f"  현재 속도 배율: {speed:.2f}x")
    if state != "PLAYING":
        print("  주의: 연주 제어는 PLAYING 상태에서만 동작합니다.")

    step = 0.1
    # 서버가 보고한 실제 배율을 기준으로 시작 (없으면 1.0 가정)
    cur_scale = speed if speed is not None else 1.0

    print("\n  명령: pause=일시정지(재개 가능) / stop=연주 중지(재개 불가)")
    print("        f=빠르게(+0.1) / d=느리게(-0.1) / s=배율 직접 입력 / q=메뉴로 복귀")

    def send_speed(target):
        """속도 요청 후 서버 실제값을 다시 읽어 표시."""
        send(s, f"PLAY_CTRL|speed|{target:.2f}")
        st, _, sp, _ = get_status(s)
        if sp is not None:
            note = " (서버 제한 적용됨)" if abs(sp - target) > 1e-6 else ""
            print(f"  속도 배율: {sp:.2f}x{note}")
            return sp
        # 속도 회신을 못 받은 경우 요청값을 그대로 사용
        print(f"  속도 배율 요청: {target:.2f}x (서버 회신 없음)")
        return target

    while True:
        cmd = input("연주 제어 (pause / stop / f / d / s / q) > ").strip().lower()

        if cmd in ("q", "quit"):
            return

        if cmd == "pause":
            send(s, "PAUSE")
            print("  일시정지 요청을 보냈습니다. (메뉴의 재개 항목으로 이어서 연주)")
            return   # 정지 후 곧 IDLE 로 돌아가므로 메뉴 복귀

        if cmd == "stop":
            send(s, "PLAY_CTRL|stop")
            print("  연주 중지 요청을 보냈습니다. (재개 지점은 폐기됩니다)")
            return   # 중지 후 곧 IDLE 로 돌아가므로 메뉴 복귀

        if cmd in ("f", "d"):
            target = round(cur_scale + (step if cmd == "f" else -step), 2)
            cur_scale = send_speed(target)
            continue

        if cmd == "s":
            raw = input("  배율(0.5~2.0) > ").strip()
            try:
                val = float(raw)
            except ValueError:
                print("    숫자를 입력하세요.")
                continue
            cur_scale = send_speed(round(val, 2))
            continue

        print("  stop / f / d / s / q 중 하나를 입력하세요.")


# 모드(메뉴) 정의 — 번호: (라벨, 보낼 OPCODE)
MODES = {
    "1": ("연주 모드 (악보)",      lambda: f"PLAY|{input('  연주 ID: ').strip()}"),
    "2": ("제스처 모드",           lambda: f"GESTURE|{input('  종류(nod/shake/wave/hi/hurray/happy): ').strip()}"),
    "3": ("단일 드럼 타격",        lambda: f"HIT|{input('  타겟(snare/ride/bass...): ').strip()}"),
    "4": ("시선 제어",             lambda: f"LOOK|{input('  pan: ').strip()}|{input('  tilt: ').strip()}"),
    "5": ("포즈 이동",             lambda: f"POSE|{input('  포즈(home/ready/shutdown): ').strip()}"),
    "9": ("궤적 재생 (CSV 오픈루프, Stage A)", lambda: f"REPLAY|{input('  CSV 절대경로: ').strip()}"),
}

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print(f"Connected to {HOST}:{PORT}")

        # 1단계: START
        while input("'start'를 입력하세요 > ").strip().lower() != "start":
            print("  start 를 입력해야 합니다.")
        send(s, "START")

        # 2단계: READY
        while input("키를 모두 제거한 후 'ready'를 입력하세요 > ").strip().lower() != "ready":
            print("  ready 를 입력해야 합니다.")
        send(s, "READY")
        print("준비 완료.")

        # 3단계: 모드 선택 루프
        while True:
            print("\n=== 모드 선택 ===")
            for k, (label, _) in MODES.items():
                print(f"  {k}. {label}")
            print("  6. 테스트 모드 (관절각 직접 입력)")
            print("  7. 연주 제어 (일시정지 / 중지 / 속도, PLAYING 중)")
            print("  8. 연주 재개 (일시정지한 곡을 멈춘 마디부터)")
            print("  q. 종료")
            choice = input("선택 > ").strip().lower()

            if choice in ("q", "quit"):
                send(s, "QUIT")
                break
            if choice == "6":
                test_mode(s)
                continue
            if choice == "7":
                play_ctrl_mode(s)
                continue
            if choice == "8":
                # 저장된 재개 지점(PAUSE로 멈춘 기록)이 있을 때만 RESUME 전송
                state, _, _, pause = get_status(s)
                if pause is None:
                    print("  저장된 재개 지점이 없습니다. (연주 중 PAUSE로 멈춘 경우에만 재개 가능)")
                    continue
                if state != "IDLE":
                    print(f"  현재 상태({state})에서는 재개할 수 없습니다. IDLE에서 다시 시도하세요.")
                    continue
                send(s, "RESUME")
                print(f"  재개 요청: 곡 {pause['id']}, 마디 {pause['bar']}부터 이어서 연주합니다.")
                continue
            if choice in MODES:
                send(s, MODES[choice][1]())   # 해당 OPCODE 생성 후 전송
            else:
                print("  올바른 번호를 선택하세요.")

if __name__ == "__main__":
    main()