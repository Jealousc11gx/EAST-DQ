# Note: EAST-DQ Basic Usage

In this note, we introduce how to use EAST-DQ , when we have device parameters. 

```bash
./quant_snn.sh <command_1 command_2>
```

Where `command_1` includes four categories of commands:

- `2d` : Visualize 2D plots, such as weight transition, origin/quantized weights and membrane potential. And it also include the computing of the write energy comsumption.
- `3d` : Visualize 3D plots, such as weight state transition.
- `ws`: Computing of the W*S operation.
- `None` : Train the SNN with varies settings.

Where `command_2` includes four categories of commands:

- `softlif`: LIF neuron with soft reset
- `hardlif`: LIF neuron with hard reset
- `softif`: IF neuron with soft reset
- `hardif`: IF neuron with hard reset

## Default Configuration

- Network Architecture: VGG8

- LIF Neuron Decay Factor: 0.5

- Encoding Method: Direct encoding

- Time Steps (T): 4

- Threshold Voltage: 1.0

- Quantization Bits: 2

 

## Parameter Description

### Parameters

- `-wq`: Enable weight quantization

- `-uq`: Enable membrane potential quantization

- `-share`: Enable scaling factor sharing

- `-sft_rst`: Enable soft reset mechanism

- `--leak_mem`: Set decay factor for LIF neurons (default: 0.5)

- `--T`: Set number of time steps for direct encoding (default: 4)

- `--th`: Set threshold voltage value (default: 1.0)

- `-quant`: Set quantization bits (default: 2)

## How to Modify Parameters

Multiple Parameter Changes:

```bash

./quant_snn.sh hardif --T 6 --th 0.8 -quant 3

```

## Examples

1. Train and analyze accuracy:
```bash
./quant_snn.sh softlif
```

2. Generate 2D visualizations:
```bash
./quant_snn.sh 2d-softlif
```

3. Create 3D visualizations:
```bash
./quant_snn.sh 3d-softlif
```

4. Analyze weight states:
```bash
./quant_snn.sh ws-softlif
```




## Important Notes

- Ensure all necessary Python dependencies are installed before running the script
- Make sure the script has execution permissions (use `chmod +x quant_snn.sh` to add execution permission)
- To change the network architecture, modify the `--arch` parameter in the script, you can modify your own architecture by change the file `quant_net.py`
- The direct encoding method is used by default for input spike generation
- The visualization of the origin/quantized weights and membrane potentia’s code is in `infer.py`

 

## SNN and Quantization Settings

### Spiking Neuron Models

```python
    def forward(self, s, share, beta, bias):

        H = s + self.membrane_potential

        grad = ((1.0 - torch.abs(H - self.thresh)).clamp(min=0))
        s = (((H - self.thresh) > 0).float() - H * grad).detach() + H * grad.detach()
        if self.soft_reset:
            U = (H - s * self.thresh) * self.leak
        else:
            U = H * self.leak * (1 - s)

        self.real_membrane_potential = U
        if self.quant_u:
            if share:
                self.membrane_potential, self.quantized_membrane_potential = u_q(U, self.num_bits_u, beta)
            else:
                self.membrane_potential = b_q(U, self.num_bits_u)
        else:
            self.membrane_potential = U

        return s
```

This code is the forward method in LIFSpike class, which describe the Spiking model. LIF model is the most used in SNNs training. Its membrane potential, as the key element of a neuron’s firing behavior, is mathematically described as,

$$U_{i}^{l}[t]=\tau U_{i}^{l}[t-1]+S_{j}^{l-1}[t]$$

where 𝜏 is the constant leaky factor, values from 0 to 1. if 𝜏 < 1, this is a LIF model, if  𝜏 = 1, this is a IF model. S ~j~^𝑙−1^ [𝑡] is the input spike from presynaptic neuron 𝑗 at time 𝑡. Neuron 𝑖 integrates inputs and emits a spike when its membrane potential exceeds the firing threshold. Mathematically, the spike generation function is stated as

$$S_{i}^{l}[t]=\begin{cases}
1, & \text{if } U_{i}^{l}[t] \ge \theta, \\
0, & \text{otherwise.}
\end{cases}$$

where 𝜃 denotes the firing threshold parameter. Following each spike emission, the spiking neuron 𝑖 undergoes a reset mechanism that updates its membrane potential. The hard reset process is mathematically defined as,

$$U_i^l[t]\leftarrow U_i^l[t]\cdot\left(1-S_i^l[t]\right)$$

The hard reset will reset the membrane potetial to zero upon emitting a spike. The soft reset process will substract a threshold to the membrane potetial. And is mathematically defined as,

$$U_{i}^{l}[t]\leftarrow U_{i}^{l}[t]-𝜃S_{i}^{l}[t]$$

The hard reset mechanism where the membrane potential of neuron 𝑖 is reset to zero upon emitting a spike and remains unchanged in the absence of a spike.

### Quantization Method

```python
def w_q(w, b, alpha):  # b is the number of bits
    w = torch.tanh(w)
    w = torch.clamp(w / alpha, min=-1, max=1)
    w = w * (2 ** (b - 1) - 1)
    w_hat = (w.round() - w).detach() + w
    return w_hat * alpha / (2 ** (b - 1) - 1), alpha


def u_q(u, b, alpha):  # b is the number of bits
    u = torch.tanh(u)
    u = torch.clamp(u / alpha, min=-1, max=1)
    u = u * (2 ** (b - 1) - 1)
    u_hat = (u.round() - u).detach() + u
    return u_hat * alpha / (2 ** (b - 1) - 1), u_hat
```

The u_q and w_q method is quantization method with [MINT](), which is a SNN QAT method,  the w_q is similar to u_q, w_q return the de-quantized weights and scaling factor alpha, you can compute the interger voltage threshold by using alpha. u_q return de-quantized and quantized values, you can reach the membrane potential by this function.



### Surrogate Gradients

Training SNNs presents a distinct challenge compared to traditional ANNs and Deep Neural Networks (DNNs) due to the non-differentiable nature of the spiking (i.e. firing) mechanism. To tackle this issue, existing studies employ surrogate gradients to approximate the true gradient . In this paper, we use the  linear surrogate gradient  which is mathematically defined as,

$$H = s + V_m$$

$$\text{grad} = \max(1 - |H - \theta|, 0)$$

the code is in the `spike_related.py`

```python
H = s + self.membrane_potential
grad = ((1.0 - torch.abs(H - self.thresh)).clamp(min=0))
```



## SNN Visualization and Power Analysis Tool

### 1. Visualization

- This function creates pie charts to visualize the distribution of weight state transitions

```python
def plot_transition_pie(weight_tracking, segment_name):
    """
    Create pie charts showing state transition distributions.
    
    Features:
    - Individual batch transitions
    - Combined segment analysis
    - Layer-wise comparison
    """
```


- This function generates averaged transition statistics in pie chart format in a certain stage

```python
def plot_average_transition_pie(weight_tracking, segment_name):
    """
    Generate average state transition visualization.
    
    Features:
    - Segment-wise analysis
    - Percentage distribution
    - Layer-specific patterns
    """
```



- This function creates 3D visualizations showing how weight states evolve across training batches

```python
def plot_weight_state_vs_batch(weight_tracking, segment_name):
    """
    Create 3D visualization of weight state evolution.
    
    Features:
    - Temporal tracking
    - State transition visualization
    - Multiple weight trajectories
    """
```



- This function generates histograms comparing original and quantized weight distributions for each layer

```python
def visualize_weights(model, quantization_setting=''):
    """
    Visualize weight distributions before and after quantization for each layer.
    
    Features:
        - Original weight distribution visualization
        - Quantized weight distribution visualization
        - Beta value and threshold reporting
        - Percentage-based frequency representation
    """
```



- This function creates distribution plots of membrane potentials before and after quantization

```python
def visualize_membrane_potential(model, data_loader, quantization_setting='', time_steps=4):
    """
    Visualize membrane potential distributions before and after quantization.
    
    Features:
        - Real-time membrane potential tracking
        - Original vs quantized potential comparison
        - Layer-wise potential distribution analysis
        - Time-step specific visualization
    """
```



### 2. Energy Analysis

#### 2.1 Write Energy Analysis

```python
def calculate_average_power(weight_tracking, segment_name):
    """
    Calculate energy consumption based on state transitions.
    
    Features:
    - State-based energy model
    - Layer-wise analysis
    - Temporal energy tracking

    """
```

  - State transition energy calculation
  - Layer-wise energy consumption

#### 2.2 Weight Activation Energy Analysis
```python
def hook_fn(module, input, output):
    """
    Hook function for tracking Weight*Spike operations
    - Counts multiplication operations between +1/-1 weights and non-zero inputs
    - Records operation statistics for each layer
    - Used for read energy consumption calculation
    """
def save_ws_statistics(ws_statistics, output_path):
    """
    Save Weight*Spike operation statistics
    - Saves statistical data in CSV format
    - Records +1/-1 weight operation counts for each layer
    """
```

  - Spike-triggered weight access counting
  - Separate +1/-1 weight access statistics
  - Layer-wise access patterns



### 3. Configuration

- This is the tracked segments, with cifar10 dataset and batchsize = 256, you can change the track segments by redefine this:

```python
tracked_batches_segments = {
    'early': list(range(0, 10)),
    'middle': list(range(95, 105)),
    'last': list(range(180, 190))
}
```
- This is the write energy comsumption matrix with 6 different pattern, you can modify the shape and parameters of the following matrix according to your device characteristics

```python
# Pattern 1:
Write_energy_consumption_1 = {
's11': 0, 's12': 5.1, 's13': 5.8,
's21': 5.1, 's22': 0, 's23': 5.8,
's31': 5.8, 's32': 10.9, 's33': 0
}

# Pattern 2:
Write_energy_consumption_2 = {
's11': 0, 's12': 5.1, 's13': 5.8,
's21': 5.1, 's22': 0, 's23': 5.8,
's31': 10.9, 's32': 5.8, 's33': 0
}

# Pattern 3:
Write_energy_consumption_3 = {
's11': 0, 's12': 10.9, 's13': 5.8,
's21': 5.8, 's22': 0, 's23': 5.1,
's31': 5.8, 's32': 5.1, 's33': 0
}

# Pattern 4:
Write_energy_consumption_4 = {
's11': 0, 's12': 5.8, 's13': 5.1,
's21': 5.8, 's22': 0, 's23': 5.1,
's31': 5.1, 's32': 10.9, 's33': 0
}

# Pattern 5:
Write_energy_consumption_5 = {
's11': 0, 's12': 5.8, 's13': 5.1,
's21': 10.9, 's22': 0, 's23': 5.8,
's31': 5.1, 's32': 5.8, 's33': 0
}

# Pattern 6:
Write_energy_consumption_6 = {
's11': 0, 's12': 5.8, 's13': 10.9,
's21': 5.8, 's22': 0, 's23': 5.1,
's31': 5.8, 's32': 5.1, 's33': 0
}
```

- This is the read energy comsumption matrix, you can modify the shape and parameters of the following matrix according to your device characteristics

```python
# Pattern 6:
Read_energy_consumption = {'weight=-1':1.807, 'weight=0':1.812, 'weight=1':1.809}

```

