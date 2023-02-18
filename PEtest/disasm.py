'''
.text
- 프로그램 실행 코드

.data
- 읽고 쓰기가 가능한 데이터 섹션으로 초기화된 전역변수와 정적변수 등이 위치.

.rdata
- 읽기 전용 데이터 섹션으로 상수형 변수, 문자열 상수 등이 위치

.bss
- 초기화되지 않은 전역변수를 담고 있는 섹션
'''
import pefile
import distorm3


pefilePath = "" # 목표 PE 파일 경로
pefilePath = "/storages/hdd0/pt0/ProjectAlaska/Datasets/track_a_learn/learn/4808531239"
pe = pefile.PE(pefilePath) # PE File 객체
textSec = pe.sections[0] # text 섹션
disasm = distorm3.Decode(0, textSec.get_data(), distorm3.Decode64Bits) # 디스어셈블

for (offset, size, instr, hexdump) in disasm:
    print(instr)# 어셈블리 코드 출력
