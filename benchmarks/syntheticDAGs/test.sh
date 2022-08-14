#!/bin/bash
export XITAO_LAYOUT_PATH=../../ptt_layout_tx2

echo "Run with most energy efficient one:"
for((i=6;i<=6;i++))
do
    for((k=0;k<5;k++))
    do
        A57=0 DENVER=0 ./FirstEnergy 1 128 1024 0 1 20000 0 0 $i 
        sleep 1
    done
done

echo "Run with second energy efficient one:"
for((i=6;i<=6;i++))
do
    for((k=0;k<5;k++))
    do
        A57=0 DENVER=0 ./SecondEnergy 1 128 1024 0 1 20000 0 0 $i
    sleep 1
    done
done
    # echo "************************* RWSS - No Sleep - A *********************"
    # for k in 1 2
    # do
    #      A57=0 DENVER=0 ./MM_RWSS_NoS_A 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "************************* RWSS - With Sleep - A *********************"
    # for k in 1 2
    # do
    #      A57=0 DENVER=0 ./MM_RWSS_S_A 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "************************* RWSS - No Sleep - D *********************"
    # for k in 1 2
    # do
    #      A57=0 DENVER=0 ./MM_RWSS_NoS_D 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "************************* RWSS - With Sleep - D *********************"
    # for k in 1 2
    # do
    #      A57=0 DENVER=0 ./MM_RWSS_S_D 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "************************* FCAS-No Sleep (Perf) *********************"
    # for k in 1 2
    # do
    #      A57=1 DENVER=0 ./MM_LCAS_NoS 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "************************* FCAS-with Sleep (Perf) *********************"
    # for k in 1 2
    # do
    #      A57=1 DENVER=0 ./MM_LCAS_S 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "************************* FCAS-No Sleep (Cost) *********************"
    # for k in 1 2
    # do
    #      A57=0 DENVER=0 ./MM_FCAS_NoS_Cost 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "************************* FCAS-with Sleep (Cost) *********************"
    # for k in 1 2
    # do
    #      A57=0 DENVER=0 ./MM_FCAS_S_Cost 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "***************** EAS-No Sleep-No Criticality *************"
    # for k in 1 2 
    # do
    #      A57=0 DENVER=0 ./MM_EAS_NoC_NoS 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "***************** EAS - with Sleep - No Criticality *************"
    # for k in 1 2 
    # do
    #      A57=0 DENVER=0 ./MM_EAS_NoC_S 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "***************** EAS-No Sleep-with Criticality (Perf) *************"
    # for k in 1 2 
    # do
    #      A57=0 DENVER=0 ./MM_EAS_C_NoS_Perf 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    #echo "***************** EAS-with Sleep-with Criticality (Perf) *************"
    #for k in 1 2 
    #do
     #    A57=0 DENVER=0 ./MM_EAS_C_S_Perf 0 64 0 0 1 50000 0 0 $i 
      #  sleep 2
    #done

    # echo "***************** EAS-No Sleep-with Criticality (Cost) *************"
    # for k in 1 2 
    # do
    #      A57=0 DENVER=0 ./MM_EAS_C_NoS_Cost 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "***************** EAS-with Sleep-with Criticality (Cost) *************"
    # for k in 1 2 
    # do
    #      A57=0 DENVER=0 ./MM_EAS_C_S_Cost 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done
    # echo "***************** EAS-with Sleep-with Criticality (Parallelism Sensitive Test) *************"
    # for k in 1 2 
    # do
    #      A57=0 DENVER=0 ./MM_EAS_C_S_PS_critical_only 0 64 0 0 1 50000 0 0 $i 
    #     sleep 2
    # done

    # echo "***************** Cri=>Perf *************"
    # for k in 1 2 3
    # do
    #     A57=0 DENVER=0 ./MM_EAS_C_S_Perf 0 64 1024 256 2 50000 0 0 $i 
    #     sleep 2
    # done

