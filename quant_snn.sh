```bash
#!/bin/bash

train_accuracy() {
    python train_snn_acc.py $@
}

train_2d_visualization() {
    python train_snn_2d_plot.py $@
}

train_3d_visualization() {
    python train_snn_3d_plot.py $@
}

train_weight_state() {
    python train_snn_ws.py $@
}

case "$1" in
    softlif)
        train_accuracy --arch vgg8 --leak_mem 0.5 -wq -uq -share -sft_rst
        ;;
    hardlif)
        train_accuracy --arch vgg8 --leak_mem 0.5 -wq -uq -share
        ;;
    softif)
        train_accuracy --arch vgg8 -wq -uq -share -sft_rst
        ;;
    hardif)
        train_accuracy --arch vgg8 -wq -uq -share
        ;;

    2d-softlif)
        train_2d_visualization --arch vgg8 --leak_mem 0.5 -wq -uq -share -sft_rst
        ;;
    2d-hardlif)
        train_2d_visualization --arch vgg8 --leak_mem 0.5 -wq -uq -share
        ;;
    2d-softif)
        train_2d_visualization --arch vgg8 -wq -uq -share -sft_rst
        ;;
    2d-hardif)
        train_2d_visualization --arch vgg8 -wq -uq -share
        ;;

    3d-softlif)
        train_3d_visualization --arch vgg8 --leak_mem 0.5 -wq -uq -share -sft_rst
        ;;
    3d-hardlif)
        train_3d_visualization --arch vgg8 --leak_mem 0.5 -wq -uq -share
        ;;
    3d-softif)
        train_3d_visualization --arch vgg8 -wq -uq -share -sft_rst
        ;;
    3d-hardif)
        train_3d_visualization --arch vgg8 -wq -uq -share
        ;;

    ws-softlif)
        train_weight_state --arch vgg8 --leak_mem 0.5 -wq -uq -share -sft_rst
        ;;
    ws-hardlif)
        train_weight_state --arch vgg8 --leak_mem 0.5 -wq -uq -share
        ;;
    ws-softif)
        train_weight_state --arch vgg8 -wq -uq -share -sft_rst
        ;;
    ws-hardif)
        train_weight_state --arch vgg8 -wq -uq -share
        ;;

    *)
        echo "Usage: $0 {softlif|hardlif|softif|hardif|2d-softlif|2d-hardlif|2d-softif|2d-hardif|3d-softlif|3d-hardlif|3d-softif|3d-hardif|ws-softlif|ws-hardlif|ws-softif|ws-hardif}"
        exit 1
        ;;
esac
```
