# ML-Based EV Battery Temperature Prediction & Cooling Optimization

An end-to-end framework coupling machine learning forecasting (XGBoost, Random Forest, LSTM) with dynamic thermal physics simulation to optimize active battery thermal management systems (BTMS).

---

## 1. Mathematical Formulation

### 1.1 Battery Heat Generation (Bernardi Model)
Heat generation inside a lithium-ion cell comprises irreversible Joule heating and reversible entropic potential:
$$\dot{Q}_{gen} = I(t)^2 R_{int}(T, SOC, \text{cycle}) - I(t) \cdot T_{core}(t) \cdot \frac{\partial U_{oc}}{\partial T}$$

- **Internal Resistance Dynamics**:
  $$R_{int}(T, \text{cycle}) = R_{nom} \cdot \exp\left[\frac{E_a}{R_g}\left(\frac{1}{T_{core}} - \frac{1}{T_{ref}}\right)\right] \cdot (1 + \alpha \cdot \text{cycle})$$

### 1.2 Two-State Lumped Parameter Thermal Dynamics
EV battery surface temperature is measured by external thermistors, while aging and thermal degradation are driven by core temperature:
$$C_{core} \frac{dT_{core}}{dt} = \dot{Q}_{gen} - \frac{T_{core} - T_{surf}}{R_{cs}}$$
$$C_{surf} \frac{dT_{surf}}{dt} = \frac{T_{core} - T_{surf}}{R_{cs}} - \dot{Q}_{amb} - \dot{Q}_{cool}$$

Where:
- $\dot{Q}_{amb} = \frac{T_{surf} - T_{amb}}{R_{amb}}$ (Passive convection to pack enclosure)
- $\dot{Q}_{cool} = u(t) \cdot K_{cool} \cdot (T_{surf} - T_{coolant})$ (Active liquid cold plate)
- $u(t) \in [0, 1]$: Actuator duty cycle / pump flow rate

### 1.3 Fluid Pumping Power (Affinity Law)
Fluid mechanics dictating parasitic electrical consumption of the BTMS pump/fan:
$$P_{pump}(u) = P_{max} \cdot u(t)^3$$

---

## 2. Cooling Control Strategies

| Strategy | Actuation Logic | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **Reactive Hysteresis (Bang-Bang)** | $u=1.0$ if $T > 34^\circ C$, $u=0$ if $T < 29.5^\circ C$ | Simple, standard in basic thermostats | High thermal oscillations, thermal overshoot due to latency, excessive energy consumption ($u=1.0$ at peak $P_{pump}$). |
| **Closed-Loop PID** | $u(t) = K_p e + K_i \int e dt + K_d \frac{de}{dt}$ | Smooth modulation, holds steady-state | Reactive only; cannot preempt high-load acceleration bursts. |
| **ML-Predictive Look-Ahead** | $u(t) = f(\hat{T}_{t+H}, I^2)$ | **Pre-cools ahead of thermal spikes**, operates at low flow ($P \propto u^3 \implies 70\%+$ energy reduction) | Requires predictive model inference in the BMS MCU. |

---

## 3. NASA Battery Aging Dataset

The project supports the **NASA Prognostics Center of Excellence (PCoE)** Lithium-Ion Battery Aging Datasets:
- Batteries: `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat`.
- Download from: [NASA PCoE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
- **Dataset Bridge**: NASA tests were conducted in passive ambient temperature fixtures (constant 2A discharge). The pipeline extracts electrical and impedance degradation characteristics from NASA cycles and couples them to our dynamic thermal-cooling simulation engine.

---

## 4. Quickstart

### Installation
```bash
pip install -r requirements.txt
```

### Running with Synthetic Dynamic EV Cycles
```bash
python main.py --num_cycles 30
```

### Running with NASA `.mat` File
```bash
python main.py --dataset B0005.mat
```
