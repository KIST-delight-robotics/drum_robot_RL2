# Phil

> 개발 중인 드럼 연주 로봇 제어 시스템.

CAN 통신 기반으로 TMotor / Maxon / Dynamixel 모터를 제어하며, LLM(TCP)을 통해 명령을 수신합니다. 이름은 드러머 Phil Collins에서 따왔습니다.

---

## 시스템 요구사항

### 운영체제 및 권한

- **OS**: Ubuntu 22.04 LTS
- **권한**: `SCHED_FIFO` 우선순위 설정 및 CAN 포트 제어를 위해 `sudo` 실행 필요

### 빌드 환경

- **컴파일러**: g++ (C++17 지원)
- **빌드 도구**: GNU Make
- **컴파일 옵션**: `-Wall -O2 -g -std=c++17 -fPIC` (Makefile 기본값)

### 의존성 (저장소에 포함)

- **Dynamixel SDK** — `drumrobot_server/lib/dynamixel_sdk/`
- **nlohmann/json** — `drumrobot_server/lib/nlohmann/json.hpp`

### 하드웨어

- USB-CAN 어댑터 (`can0`, `can1` 등)
- Dynamixel U2D2 (`/dev/ttyUSB0`)
- USB 허브 전원 제어용 `uhubctl` — CAN 포트 리셋에 사용

---

## 운용 절차 (전원 ON ~ 연주)

이 로봇은 안전을 위해 **고정 키(locking pin)** 로 초기 위치를 잡은 뒤 단계적으로 활성화합니다. 절차를 반드시 지켜야 합니다.

1. **고정 키를 꽂은 상태로** 로봇 전원을 켭니다.
2. 코드를 실행하면 모터 통신을 확인하고 대기합니다. 이 시점에는 **모터 토크가 걸려 있지 않습니다** (`Standby`).
3. `START` 입력 → 모터 토크 ON, home 자세 유지 (`Init`). 콘솔에 고정 키 제거 안내가 출력됩니다.
4. **고정 키를 모두 제거**합니다. (토크가 걸려 자세를 붙들고 있는 상태에서 제거)
5. `READY` 입력 → 동작 허용 상태로 전환 (`Idle`). 이후 LOOK / MOVE / POSE / GESTURE / HIT / PLAY 명령을 사용할 수 있습니다.

> 고정 키를 제거하기 전(`Init` 상태)에는 팔을 움직이는 동작 명령이 모두 거부됩니다. 반드시 키 제거 후 `Idle`로 전환하세요.

---

## 시스템 구조

### 데이터 흐름

```
입력 (TCP)
        ↓
   CommandQueue        (문자열 명령)
        ↓
   CommandParser       (명령 파싱 → ParsedCommand)
        ↓
   BehaviorPlanner     (ParsedCommand → MotionPrimitive 시퀀스)
        ↓
   MotionQueue         (MotionPrimitive)
        ↓
   TrajectoryGenerator (궤적 생성 → ControlSetPoint 시퀀스)
        ↓
   ControlQueue        (5ms 주기 ControlSetPoint)
        ↓
   Controller          (CAN 송수신, 1ms / 5ms 주기)
        ↓
   CAN Bus → 모터
```

### 핵심 컴포넌트

| 컴포넌트 | 책임 |
|---|---|
| `TcpServer` | 사용자 입력(TCP)을 받아 `CommandQueue`에 푸시 |
| `CommandParser` | 문자열 명령을 파싱해 `ParsedCommand`(Opcode + args)로 변환 |
| `BehaviorPlanner` | `ParsedCommand`를 받아 `MotionPrimitive` 시퀀스 생성. 로봇 상태(`RobotState`) 전이 관리 |
| `TrajectoryGenerator` | `MotionPrimitive`를 받아 5ms 단위 `ControlSetPoint` 궤적 생성 (관절/작업공간/연주) |
| `PlayMotionGenerator` | 연주(DRUM) 모션 합성. Base / State / Pedal / Head 4개 서브 제너레이터를 IK로 통합 |
| `MotionPlanner` | 위 컴포넌트를 orchestrate하는 스레드 루프. `CommandQueue` 소비 및 `ControlQueue` 잔량 관리 |
| `KinematicsSolver` | 손 끝 좌표 → 9개 관절각 (역기구학), 9개 관절각 → 손 끝 좌표 (순기구학) |
| `Controller` | `ControlQueue`에서 꺼내 CAN 프레임 송신 (1ms Maxon 보간 + 5ms 동시), 모터 상태 수신 및 안전 검사 |
| `Robot` | 모터 객체들의 컨테이너. 초기화, socket 할당, Maxon/Dynamixel 설정 |
| `CanInterface` | CAN 포트 활성화, 소켓 생성, 프레임 read/write, Non-block 설정 |
| `MotorCodec` | TMotor / MaxonMotor 종류별 CAN 프레임 구성·해석. 명령 프레임 빌드는 `encode*`, 수신 프레임 파싱은 `decodeFeedback` |
| `Logger` | 모터/궤적/명령 상태를 타임스탬프 기반 CSV 파일로 저장 |

### 연주 모션 서브 제너레이터 (`PlayMotionGenerator`)

| 제너레이터 | 담당 관절 | 역할 |
|---|---|---|
| `BaseMotionGenerator` | 팔 (작업공간 좌표 → IK) | 악기 위치로 손 끝 이동 (Bézier 경로) |
| `StateMotionGenerator` | 팔꿈치 / 손목 보정 | 타격 강도에 따른 들어올림·내려침 디테일 |
| `PedalMotionGenerator` | 페달 (9, 10) | 베이스 드럼, 하이햇 open/closed |
| `HeadMotionGenerator` | 머리 (11, 12) | 악기 응시(yaw) + 박자 끄덕임(pitch) |

### 스레드 구조

| 스레드 | 우선순위 | 주기 | 역할 |
|---|---|---|---|
| `send_thread` | 40 | 1ms (Maxon 보간) / 5ms (TMotor + Maxon + Dynamixel) | CAN 프레임 송신 |
| `recv_thread` | 30 | 100µs | CAN 프레임 수신 및 모터 상태 갱신, 안전 검사 |
| `motion_planning_thread` | 20 | 5ms 폴링 | CommandQueue → MotionQueue → ControlQueue 변환 |
| `tcp_server_thread` | 10 | 블로킹 | 사용자 입력(TCP) 수신 |

우선순위는 `SCHED_FIFO` 정책 기준 (값이 클수록 높음).

---

## 파일 구조

```
Phil/
├── Makefile
├── drumrobot_client/
│   └── main.py                             # TCP 클라이언트 (LLM 모드용)
└── drumrobot_server/
    ├── Makefile
    ├── bin/                                # 실행 파일 (빌드 시 생성)
    ├── obj/                                # 오브젝트 파일 (빌드 시 생성)
    ├── log/                                # Logger 출력 디렉토리 (런타임 생성)
    ├── config/
    │   ├── motors.json                     # 모터 설정 (ID, 관절 범위, PDO ID 등)
    │   ├── robot_poses.json                # 사전 정의 포즈 (init / home / ready / shutdown)
    │   ├── kinematics.json                 # 관절 한계, 링크 길이
    │   ├── drum_coordinate.json            # 악기별 손 끝 좌표 및 손목 각도
    │   └── can_ports.json                  # 머신별 USB 허브/포트 매핑 (CAN 리셋용)
    ├── data/
    │   ├── midi/                           # MIDI 원본
    │   └── scores/                         # 연주용 악보 (.txt)
    │   └── audio/                          # 음악 파일 (.wav)
    ├── lib/
    │   ├── dynamixel_sdk/
    │   └── nlohmann/json.hpp
    ├── include/
    │   ├── common/                         # app_context, command/control/motion_queue, robot_config
    │   ├── hardware/                       # can_interface, motor, motor_codec, robot
    │   ├── kinematics/                     # kinematics_solver
    │   ├── tcp/                            # tcp_server, command_parser
    │   ├── realtime/                       # controller
    │   ├── trajectory/                     # behavior_planner, motion_planner,
    │   │                                   #   trajectory_generator, play/base/state/pedal/head_motion_generator
    │   └── util/                           # logger
    └── src/                                # 위 헤더에 대응하는 구현 (main.cpp 포함)
```

---

## 좌표계 / 운동학 컨벤션

### 단위

| 위치 | 단위 |
|---|---|
| 사용자 입력 (TCP) | 도 (degree) |
| JSON 설정 파일 (`motors.json`, `robot_poses.json`, `kinematics.json`) | 도 (degree) |
| `drum_coordinate.json` 의 `wrist_angle_deg` | 도 (degree, 로드 시 라디안 변환) |
| 코드 내부 (`q`, `qd`, 모든 관절각) | 라디안 (rad) |
| 작업공간 좌표 (`right_position`, `left_position`) | 미터 (m) |

### 좌표계

- 베이스 원점: 허리(waist) 회전축
- 어깨 간격: `link_length.waist` (0.520 m)
- 상완 길이: `link_length.upper_arm` (0.230 m)
- 하완 길이: `link_length.forearm` (0.200 m)
- 스틱 길이: `link_length.stick` (0.373 m)

값은 `kinematics.json` 참조.

### DH 컨벤션

- Modified DH (Craig) 사용
- 변환 행렬: `KinematicsSolver::dh_transform(a, alpha, d, theta)`
- 오른팔 / 왼팔 각각 6단 DH로 손 끝 좌표 계산

### 관절 방향

- 각 관절의 회전 방향은 `motors.json`의 `direction_sign` (±1.0)로 정의
- 모터 좌표 → 관절 좌표: `joint = motor * direction_sign + initial_joint_angle`

---

## 모터 구성

총 13개 관절. `motors.json`에 ID 순서대로 정의됨.

| ID | 이름 | 타입 | 모델 |
|---|---|---|---|
| 0  | waist            | TMotor     | AK10-9       |
| 1  | right_shoulder_1 | TMotor     | AK70-10      |
| 2  | left_shoulder_1  | TMotor     | AK70-10      |
| 3  | right_shoulder_2 | TMotor     | AK70-10      |
| 4  | right_elbow      | TMotor     | AK70-10      |
| 5  | left_shoulder_2  | TMotor     | AK70-10      |
| 6  | left_elbow       | TMotor     | AK70-10      |
| 7  | right_wrist      | MaxonMotor | DCX22L       |
| 8  | left_wrist       | MaxonMotor | DCX22L       |
| 9  | right_pedal      | MaxonMotor | DCX32L       |
| 10 | left_pedal       | MaxonMotor | DCX32L       |
| 11 | head_yaw         | Dynamixel  | XM430-W210-T |
| 12 | head_pitch       | Dynamixel  | XM430-W210-T |

### 지원 제어 모드

| 모터 | 제어 모드 |
|---|---|
| TMotor       | `POS` (SET_POS), `VEL` (SET_RPM + P 피드백), `SET_POS_SPD`, `SET_ORIGIN`, `CURRENT_BRAKE` |
| TMotor (MIT) | Position-Velocity-Torque 통합 코덱 구현됨. 단, Controller에서 현재 미사용 |
| MaxonMotor   | `CSP` (Cyclic Sync Position), `CST` (Cyclic Sync Torque), `HMM` (Homing). `CSV` 미구현 |
| Dynamixel    | 위치 제어 (Profile Acceleration / Velocity + Goal Position) |

### 제어 모드 기본값 (`trajectory_generator.hpp`)

- TMotor(0~6) = `VEL`
- Wrist(7, 8) = `CSP` (단, 연주 중에는 `CST`로 전환)
- Pedal(9, 10) = `CSP`
- Head(11, 12) = Dynamixel 위치 제어 (`ControlMode::NONE`)

### 모터 통신

| 모터 | 물리 계층 | 프로토콜 |
|---|---|---|
| TMotor      | CAN 1Mbps                    | TMotor 독자 Servo / MIT 모드 |
| MaxonMotor  | CAN 1Mbps                    | CANopen (SDO / PDO) |
| Dynamixel   | UART 4.5Mbps (`/dev/ttyUSB0`) | Dynamixel Protocol 2.0 |

---

## 명령 프로토콜

### 패킷 형식

```
OPCODE|arg1|arg2\n
```

필드 구분자 `|`, 패킷 구분자 `\n`. Opcode는 대소문자를 구분하지 않습니다.

### Opcode 목록

| Opcode | 인자 | 설명 | 허용 상태 |
|---|---|---|---|
| `START`     | 없음                                            | 모터 토크 ON + home 자세. `Standby → Init` | Standby |
| `READY`     | 없음                                            | 고정 키 제거 완료 후 동작 허용 상태로 전환. `Init → Idle` | Init |
| `LOOK`      | `pan_deg`, `tilt_deg`                          | 머리 yaw / pitch 제어 | Idle |
| `GESTURE`   | `type`                                          | 제스처 (`nod` / `shake` / `wave` / `hi` / `hurray` / `happy`) | Idle |
| `MOVE`      | `motor_name`, `angle_deg`, `[move_time=3.0]`   | 개별 관절 이동 | Idle |
| `POSE`      | `pose_name`                                     | 사전 정의 포즈로 이동 (`home` / `ready` / `shutdown`) | Idle |
| `HIT`       | `target`                                        | 단일 드럼 타격 | Idle |
| `PLAY`      | `id`                                            | 악보 연주. `config/play_list.json`의 id로 악보/음원 선택 | Idle |
| `PAUSE`     | 없음                                            | 연주 일시정지. **재개 지점(마디) 저장** 후 ready 복귀 → `RESUME` 가능 | Playing |
| `RESUME`    | 없음                                            | 일시정지한 곡을 저장된 마디부터 재개 (음악 없이 무음 재개) | Idle |
| `PLAY_CTRL` | `stop` 또는 `speed`, `scale`                    | 연주 제어. `stop`=완전 중지(재개 지점 폐기), `speed`=속도 배율(0.5~2.0) | Playing |
| `GET_STATUS`| 없음                                            | 상태 조회. 응답: `STATUS\|<state>\|<q0..q12>\|<speed>\|<pause_valid>\|<pause_id>\|<pause_bar>` (`pause_valid=1`이면 RESUME 가능) | 모든 상태 |
| `QUIT` / `Q`| 없음                                            | shutdown 포즈 이동 후 시스템 종료 | 모든 상태 |

> `START` 와 `QUIT` 외의 명령은 `send_active`가 켜진 이후(=`START`로 첫 궤적이 적재된 이후)에만 처리됩니다.

### HIT target 목록

`snare`, `floor`, `mid`, `top`, `closed hihat`, `open hihat`, `ride`, `right crash`, `left crash`, `bass`

> 타깃 문자열은 코드의 `instrument_name_to_id`(`include/common/robot_config.hpp`) 키와 정확히 일치해야 합니다(공백 포함). 예: `closed hihat`, `right crash`. 일치하지 않으면 `Unknown target instrument`로 거부됩니다.

### 명령 예시

```
START                       # 토크 ON + home (이후 고정 키 제거)
READY                       # 고정 키 제거 후 동작 허용 상태로 전환
POSE|ready                  # ready 포즈로 이동
LOOK|0|10                   # 정면, 위쪽 10도 응시
LOOK|-30|0                  # 왼쪽 30도 응시
GESTURE|nod                 # 끄덕임
GESTURE|wave                # 손 흔들기
MOVE|right_wrist|45         # right_wrist를 45도로 이동 (기본 3.0초)
MOVE|right_wrist|45|1.0     # right_wrist를 45도로 1.0초에 이동
MOVE|waist|-30|2.0          # 허리를 -30도로 2초에 이동
HIT|snare                   # 스네어 1회 타격
HIT|closed hihat            # 클로즈드 하이햇 1회 타격 (타깃 문자열에 공백 포함)
PLAY|BF                     # play_list.json의 BF(BasicFillin) 연주
PAUSE                       # 연주 일시정지 (재개 지점 저장, PLAYING 중)
RESUME                      # 멈춘 마디부터 이어서 연주 (IDLE 중)
PLAY_CTRL|stop              # 연주 완전 중지 (재개 지점 폐기)
PLAY_CTRL|speed|1.2         # 연주 속도 1.2배
QUIT                        # shutdown 포즈 이동 후 종료
```

각도 단위는 모두 **도(degree)** 이며 내부적으로 라디안으로 변환됩니다.

### 사전 정의 포즈 (`robot_poses.json`)

| 포즈 이름 | 설명 |
|---|---|
| `init`     | 초기 자세. TMotor / Maxon(관절 0~10)은 `motors.json`의 `initial_joint_angle`과 일치 |
| `home`     | 연주 대기 자세 (`START` 시 이동, 고정 키 위치) |
| `ready`    | 준비 자세 |
| `shutdown` | 종료 전 안전 자세 |

---

## 로봇 상태 (RobotState)

상태는 `AppContext::robot_state`(초기값 `Standby`)에 보관되며, `BehaviorPlanner`가 명령을 처리하면서 전이시킵니다.

```
Standby ──START──▶ Init ── READY ──▶ Idle ──PLAY──▶ Playing
   │                 │                 │               │
   │                 │                 │   연주 종료    │
   │                 │                 ◀───────────────┘
   │                 │                 │
   └──────────── QUIT / shutdown 포즈 ──┴──────────────▶ ShuttingDown
```

| 상태 | 의미 |
|---|---|
| `Standby`      | 전원 ON, 통신 확인 완료, **토크 미인가** 대기 상태. `START` 대기 |
| `Init`         | 토크 ON, home 자세 유지. **고정 키 제거** 후 `READY` 대기 |
| `Idle`         | 동작 허용. 모든 동작 명령 수신 가능 |
| `Playing`      | 연주 중 (`PLAY` 진입 시) |
| `ShuttingDown` | 종료 절차. 새 명령 거부, ControlQueue 소진 후 시스템 종료 |

### 상태 전이 트리거

| 전이 | 트리거 | 처리 위치 |
|---|---|---|
| `Standby → Init`      | `START` 입력 (`handle_start`). home 자세로 이동시키고 토크 ON | `BehaviorPlanner` |
| `Init → Idle`         | `READY` 입력 (`handle_ready`). 고정 키 제거 완료 후 동작 허용 | `BehaviorPlanner` |
| `Idle → Playing`      | `PLAY` 입력 (`handle_play`) 또는 `RESUME` 입력 (`handle_resume`) | `BehaviorPlanner` |
| `Playing → Idle`      | 악보 소진 후 motion_queue가 비면 자동 복귀 (`schedule_idle_motion`). `PAUSE`/`PLAY_CTRL\|stop`도 잔여 모션 폐기 후 같은 경로로 복귀 | `MotionPlanner` |
| `* → ShuttingDown`    | `QUIT` 입력 또는 `POSE shutdown` | `BehaviorPlanner` |

> `START` / `QUIT` 외의 명령은 `send_active`가 켜진 이후(=`START`로 첫 궤적이 적재된 이후)에만 처리됩니다. 또한 동작 명령(`LOOK` / `GESTURE` / `MOVE` / `POSE` / `HIT` / `PLAY`)은 `Idle` 상태에서만 허용되며, 그 외 상태에서는 거부됩니다.

### 상태별 대기(filler) 모션

`MotionPlanner`는 명령이 없고 `motion_queue`가 비어 있을 때(`schedule_idle_motion`), 현재 상태에 맞는 대기 모션을 채워 넣어 제어 주기를 유지합니다. 대기 모션의 종류는 `MotionType` enum으로 구분됩니다.

| 상태 | 채워지는 `MotionType` | 동작 |
|---|---|---|
| `Init`         | `STANDBY` | 고정 키 제거 전 **현재 위치 유지** (자세를 붙들고 대기) |
| `Idle`         | `IDLE`    | 대기 동작 |
| `Playing`      | (악보 소진 시) `Idle`로 전이 후 `IDLE` | 연주 종료 처리 |
| `ShuttingDown` | 없음      | 대기 모션을 채우지 않고 ControlQueue 소진을 기다림 |

> `MotionType::STANDBY`는 `Init` 상태의 대기 모션 전용 타입입니다. `MotionType::IDLE`은 `Idle` / `Playing` 종료 시 사용됩니다.

---

## 연주(PLAY) 처리

### 악보 형식 (`data/scores/<name>.txt`)

탭(`\t`) 구분 텍스트. 첫 줄은 `bpm`, 마지막은 `end`, 사이는 타격 이벤트입니다.

```
bpm	100
<bar>	<beat>	<note_R>	<note_L>	<vel_R>	<vel_L>	<is_kick>	<is_closed_hihat>	...
...
end
```

| 열 | 의미 |
|---|---|
| `bar`              | 마디 번호 |
| `beat`             | 직전 이벤트로부터의 박자 간격 (예: 0.6 = 한 박) |
| `note_R` / `note_L`| 오른팔 / 왼팔 타격 악기 번호 (0 = 무타격) |
| `vel_R` / `vel_L`  | 오른팔 / 왼팔 타격 강도 |
| `is_kick`          | 베이스 드럼 (1/0) |
| `is_closed_hihat`  | 하이햇 닫음 (1/0) |

### 악기 번호 ↔ 좌표

악기 번호는 `drum_coordinate.json`의 `id`와 매칭됩니다.

| id | 악기 | id | 악기 |
|---|---|---|---|
| 1 | snare | 5 | hihat |
| 2 | floor | 6 | ride |
| 3 | mid   | 7 | crashR |
| 4 | top   | 8 | crashL |

각 악기는 오른팔/왼팔 각각의 `position`(m)과 `wrist_angle_deg`를 가집니다.

### 시퀀스 구성

`PLAY`는 다음 순서로 `MotionPrimitive` 시퀀스를 생성합니다.

1. `DRUM(START)` — 연주 시작 자세(스네어 대기)로 이동
2. `DRUM(PLAYING)` × N — 슬라이딩 윈도우로 악보를 구간별로 잘라 생성. 한 마디(100 bpm 기준 2.4초) 분량이 모이면 한 구간을 방출
3. `DRUM(END)` — home 자세로 복귀

---

## 궤적 프로파일

`TrajectoryGenerator`가 지원하는 보간 방식. `make_translate`의 기본값은 `COSINE`, `POSE`는 `TRAPEZOIDAL`.

| 프로파일 | 설명 |
|---|---|
| `COSINE`       | 사인 보간. 양 끝에서 속도 0, 부드러운 가속 / 감속 (기본값) |
| `CUBIC`        | 3차 다항식. 양 끝에서 속도 0 |
| `QUINTIC`      | 5차 다항식. 양 끝에서 속도·가속도 모두 0 |
| `TRAPEZOIDAL`  | 사다리꼴 속도 프로파일. 가속(25%) → 등속(50%) → 감속(25%) |

---

## 상태 플래그 (AppContext)

| 플래그 | 의미 |
|---|---|
| `running`     | 전체 종료 플래그. false가 되면 모든 스레드 루프 탈출 |
| `send_active` | `Controller::send_loop` 활성화 신호. `MotionPlanner`가 첫 궤적을 `ControlQueue`에 적재할 때 켜짐 |
| `recv_active` | `Controller::recv_loop` 활성화 신호. `send_active`와 동일 시점에 켜짐 |
| `robot_state` | 위 RobotState. 초기값 `Standby` |

---

## 안전 메커니즘

### TMotor 전류 초과 차단

- 수신된 `current_motor_current`가 `current_limit`을 **연속 5회 초과**하면 시스템 정지
- 일시적 과전류(스파이크)는 카운터가 리셋되어 무시됨 (`Motor::cnt`)

### IK 실패 처리

- `KinematicsSolver::ik_solve()`가 실패를 반환하면 해당 모션을 폐기하고 시스템은 계속 동작
- 연주 중 IK 실패 시 해당 구간 생성을 중단함
- 실패 원인: 도달 불가능한 좌표, NaN / Inf, 관절 한계 초과

### 송신 전 안전 검사 (TMotor)

- **급변 차단**: 목표 위치와 현재 위치 차이가 **10도(약 0.175 rad)** 이상이면 해당 모터 송신 건너뜀
- **범위 검사**: `motors.json`의 `min_angle` / `max_angle`을 벗어나면 송신 건너뜀
- MaxonMotor / Dynamixel은 송신 전 검사 없음 (모터 자체 제한에 의존)

### 수신 후 안전 검사

- **TMotor 범위 초과** → 즉시 `running = false` (전체 정지)
- **Maxon 범위 초과** → `encodeShutdown` PDO 송신 후 전체 정지

복구 절차:

1. 시스템 종료 후 모터를 안전 범위로 수동 이동
2. CAN 포트 재기동 (USB 전원 차단 후 재연결)
3. `motors.json` 한계 값 검토 후 재실행

---

## 빌드 및 실행

### 빌드

```bash
make
```

### 실행

TCP 서버로 실행됩니다 (포트 1951):

```bash
sudo ./drumrobot_server/bin/main.out
```

또는 최상위 Makefile 사용:

```bash
make run
```

### TCP 클라이언트 (LLM 모드, 별도 터미널)

```bash
python3 drumrobot_client/main.py
```

호스트 `127.0.0.1`, 포트 `1951`로 연결됩니다.

---

## 참고 사항

- **CAN bitrate**: 1Mbps
- **CAN 포트 리셋**: `can_ports.json`에 등록된 머신에서 `uhubctl`로 USB 허브 전원을 껐다 켭니다. 미등록 머신은 리셋을 건너뜁니다.
- **Maxon 보간**: `send_loop`에서 5ms 구간을 1ms × 5 스텝으로 분할해 CSP 위치를 선형 보간 전송. 5ms 시점에 TMotor / Maxon / Dynamixel 동시 송신
- **`virtual_maxon_motor`**: 소켓당 Maxon 모터 1개를 대표로 선정해 Sync 프레임(0x80) 전송에 사용
- **Logger**: 런타임에 `drumrobot_server/log/`에 `log_MMDD_HHmm_{name}.csv` 생성 (motor / trajectory / motion_command)

---

## 라이선스 및 외부 의존성

> ⚠️ 현재 저장소에 LICENSE 파일이 없습니다. 공개 전 라이선스 명시를 권장합니다.

| 라이브러리 | 위치 | 라이선스 |
|---|---|---|
| nlohmann/json   | `drumrobot_server/lib/nlohmann/json.hpp` | MIT        |
| Dynamixel SDK   | `drumrobot_server/lib/dynamixel_sdk/`    | Apache 2.0 |