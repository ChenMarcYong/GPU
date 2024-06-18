CC = nvcc
NAME = prog

TXT_FILE_MCU = $(wildcard *.cu)
TXT_FILE_CPP1 = $(wildcard *.cpp)
TXT_FILE_CU1 = $(patsubst main.cu,,$(TXT_FILE_MCU))

TXT_FILE_CU2 = $(wildcard */*.cu)
TXT_FILE_CPP2 = $(wildcard */*.cpp)

TXT_FILE_CU3 = $(wildcard */*/*.cu)
TXT_FILE_CPP3 = $(wildcard */*/*.cpp)


TXT_FILE_CU = $(TXT_FILE_CU1) $(TXT_FILE_CU2) $(TXT_FILE_CU3)
TXT_FILE_CPP = $(TXT_FILE_CPP1) $(TXT_FILE_CPP2) $(TXT_FILE_CPP3)



all: $(TXT_FILE_CU) $(TXT_FILE_CPP)
	@$(CC) -o $(NAME) main.cu $(TXT_FILE_CPP) $(TXT_FILE_CU)
	@echo "\033[32m\tCompilation OK\033[39m"

clean :
	rm -rf $(NAME)
	@echo "\033[32m\tNettoyage OK\033[39m"