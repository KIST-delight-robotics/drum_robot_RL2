# ===== 컴파일러 설정 =====
CC = g++

# ===== 컴파일 옵션 =====
CFLAGS = -Wall -O2 -g -std=c++17 -fPIC

# ===== 헤더 파일 경로 =====
INCLUDE = -I./include -I./lib -I./lib/dynamixel_sdk/include

# ===== 링크할 라이브러리 =====
# (멀티스레드) (수학) (동적 로딩)
LDFLAGS = -lpthread -lm -ldl

# ===== 경로 변수 =====
SRCDIR = ./src
OBJDIR = ./obj
BINDIR = ./bin
EXECUTABLE = main.out

# ===== 소스/오브젝트 파일 자동 탐색 =====
SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
OBJFILES := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))

DXL_SOURCES := $(shell find ./lib/dynamixel_sdk/src -maxdepth 1 -name '*.cpp')
DXL_OBJECTS := $(patsubst ./lib/dynamixel_sdk/src/%.cpp, $(OBJDIR)/dxl/%.o, $(DXL_SOURCES))

# ===== 기본 빌드 타깃 =====
all: directories $(BINDIR)/$(EXECUTABLE)

# ===== 폴더 생성 =====
directories:
	mkdir -p $(BINDIR)
	mkdir -p $(OBJDIR)/common $(OBJDIR)/hardware $(OBJDIR)/kinematics $(OBJDIR)/tcp $(OBJDIR)/realtime $(OBJDIR)/trajectory $(OBJDIR)/util
	mkdir -p $(OBJDIR)/dxl

# ===== 링크: .o 파일들 → 실행파일 =====
$(BINDIR)/$(EXECUTABLE): $(OBJFILES) $(DXL_OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDE) $(LDFLAGS)

# ===== 컴파일: src/*.cpp → .o =====
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDE)

# ===== 컴파일: Dynamixel SDK *.cpp → .o =====
$(OBJDIR)/dxl/%.o: ./lib/dynamixel_sdk/src/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@ $(INCLUDE)

# ===== 실행 =====
# config/log 경로가 모두 'drumrobot_server/...' 상대경로라, 그 경로가 보이는 위치에서 실행한다.
# 1) 프로젝트 루트(Phil-drum-robot)에서 바로 보이면 그대로 실행
# 2) 안 보이면 상위 디렉터리로 올라가 실행 (drumrobot_server 안에서 make run 한 경우)
run: all
	@if [ -f drumrobot_server/config/motors.json ]; then \
		sudo ./drumrobot_server/bin/$(EXECUTABLE); \
	elif [ -f ../drumrobot_server/config/motors.json ]; then \
		cd .. && sudo ./drumrobot_server/bin/$(EXECUTABLE); \
	else \
		echo "[run] drumrobot_server/config/motors.json 를 찾을 수 없습니다"; exit 1; \
	fi

# ===== 청소 =====
clean:
	rm -rf $(OBJDIR) $(BINDIR)