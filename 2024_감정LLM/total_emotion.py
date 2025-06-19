import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from BLOOM2_main_original_onlyllm.interact_gguf import main as main1
from BLOOM2_main_record_S_N.interact_gguf import main as main2
from BLOOM2_main_record_V_S_N.interact_gguf import main as main3

if __name__ =="__main__":
    while True:
        version = input("Insert the number of version(0:LLM / 1:LLM+Speech / 2:LLM + Vision + Speech): ")
        if version == '0': #only LLM
            main1()
            break
        elif version == '1': #LLM + speech
            main2()
            break
        elif version == '2': #LLM + vision + speech
            main3()
            break
        elif version == "quit":
            break
        else:
            print("Please enter the correct version.")
            continue