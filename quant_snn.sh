#!/bin/bash

train_and_shutdown() {
    python train_snn.py $@
}

case "$1" in
    softlif)
        train_and_shutdown --arch vgg8 --leak_mem 0.5 -wq -uq -share -sft_rst
        ;;
    hardlif)
        train_and_shutdown --arch vgg8 --leak_mem 0.5 -wq -uq -share
        ;;
    softif)
        train_and_shutdown --arch vgg8 -wq -uq -share -sft_rst
        ;;
    hardif)
        train_and_shutdown --arch vgg8 -wq -uq -share
        ;;
    *)
        echo "Usage: $0 {softlif|hardlif|softif|hardif}"
        exit 1
        ;;
esac
~
~
~
~
~
