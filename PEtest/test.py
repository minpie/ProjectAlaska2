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


def getTextSection(peObj):
    # PE 파일 .text 섹션 내용 반환
    for i in peObj.sections:
        if (i.Name.decode("utf-8").rstrip("\x00") == ".text"):
            return i.get_data()

def getDataSection(peObj):
    # PE 파일 .data 섹션 내용 반환
    for i in peObj.sections:
        if (i.Name.decode("utf-8").rstrip("\x00") == ".data"):
            return i.get_data()

def getRdataSection(peObj):
    # PE 파일 .rdata 섹션 내용 반환
    for i in peObj.sections:
        if (i.Name.decode("utf-8").rstrip("\x00") == ".rdata"):
            return i.get_data()

def getBssSection(peObj):
    # PE 파일 .bss 섹션 내용 반환
    for i in peObj.sections:
        if (i.Name.decode("utf-8").rstrip("\x00") == ".bss"):
            return i.get_data()

def getPeObject(peFilePath):
    # pefile 객체 반환
    return pefile.PE(peFilePath)


pe = getPeObject("/storages/hdd0/pt0/ProjectAlaska/Datasets/track_a_learn/learn/4808531239")


textSec = getTextSection(pe)


disasm = distorm3.Decode(0, textSec, distorm3.Decode64Bits)


for (offset, size, instr, hexdump) in disasm:
    print(instr)
