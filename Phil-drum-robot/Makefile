all:
	$(MAKE) -C drumrobot_server

clean:
	$(MAKE) -C drumrobot_server clean

# 음악 서버 통로, 작업 폴더, 쿠키
run:
	$(MAKE) -C drumrobot_server
	sudo PULSE_SERVER=unix:/run/user/$(shell id -u)/pulse/native \
	     XDG_RUNTIME_DIR=/run/user/$(shell id -u) \
	     PULSE_COOKIE=$(HOME)/.config/pulse/cookie \
	     ./drumrobot_server/bin/main.out