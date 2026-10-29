"""
Battery Thermal & Cooling Actuation Physics Model
-------------------------------------------------
Implements a 2-State (Core and Surface) Lumped Parameter Thermal Network
coupled with an active liquid/air cooling actuator.

Physics equations:
1. Core Thermal Balance:
   C_core * d(T_core)/dt = Q_gen - (T_core - T_surf) / R_cs
2. Surface Thermal Balance:
   C_surf * d(T_surf)/dt = (T_core - T_surf) / R_cs - Q_amb - Q_cool
3. Heat Generation (Bernardi Formulation):
   Q_gen = I^2 * R_int(T_core, SOC, cycle) + I * T_core * dU_dT
4. Active Cooling:
   Q_cool = u * h_cool_max * A_cool * (T_surf - T_coolant)
5. Parasitic Pumping Power (Affinity Laws):
   P_pump(u) = P_max * u^3
"""

import numpy as np

class BatteryThermalModel:
    def __init__(
        self,
        c_core: float = 38.0,        # Core heat capacity (J/K) ~ 18650/21700 cell
        c_surf: float = 8.0,         # Surface heat capacity (J/K)
        r_cs: float = 1.8,           # Thermal resistance between core and surface (K/W)
        r_amb: float = 12.0,         # Passive thermal resistance to ambient (K/W)
        r_int_nom: float = 0.045,    # Nominal electrical internal resistance (Ohms)
        du_dt: float = -0.00022,     # Entropic coefficient (V/K)
        t_amb: float = 24.0,         # Ambient temperature (deg C)
        t_coolant: float = 20.0,     # Coolant inlet temperature (deg C)
        k_cool_max: float = 2.5,     # Max active cooling conductance (W/K)
        p_pump_max: float = 45.0,    # Maximum pump electrical power (Watts per module/pack fraction)
    ):
        self.c_core = c_core
        self.c_surf = c_surf
        self.r_cs = r_cs
        self.r_amb = r_amb
        self.r_int_nom = r_int_nom
        self.du_dt = du_dt
        self.t_amb = t_amb
        self.t_coolant = t_coolant
        self.k_cool_max = k_cool_max
        self.p_pump_max = p_pump_max

        # States (in deg C)
        self.t_core = t_amb
        self.t_surf = t_amb

    def reset(self, initial_temp: float = 24.0):
        """Reset thermal states."""
        self.t_core = float(initial_temp)
        self.t_surf = float(initial_temp)

    def compute_internal_resistance(self, soc: float = 0.5, cycle: int = 1) -> float:
        """
        Internal resistance varies with core temperature (Arrhenius), SOC, and cycle aging.
        """
        # Temperature dependence: R rises at low temperatures
        t_kelvin = self.t_core + 273.15
        t_ref = 298.15 # 25 deg C
        arrhenius_factor = np.exp(1200.0 * (1.0 / t_kelvin - 1.0 / t_ref))
        
        # SOC dependence: R rises at very low (<10%) and very high (>90%) SOC
        soc_factor = 1.0 + 0.15 * (1.0 - soc)**2 + 0.1 * (soc > 0.9)
        
        # Aging degradation: ~0.05% resistance increase per cycle
        aging_factor = 1.0 + 0.0005 * cycle
        
        return self.r_int_nom * arrhenius_factor * soc_factor * aging_factor

    def compute_heat_generation(self, current: float, soc: float = 0.5, cycle: int = 1) -> float:
        """
        Calculates instantaneous heat generation Q_gen (Watts) using Bernardi equation.
        I > 0 for discharge, I < 0 for charge.
        """
        r_int = self.compute_internal_resistance(soc=soc, cycle=cycle)
        # Joule heating: I^2 * R (always positive)
        q_joule = (current ** 2) * r_int
        
        # Entropic heating (reversible): - I * T * (dU/dT)
        # Note: during discharge (I > 0) with dU/dT < 0, this term is positive (adds heat)
        t_kelvin = self.t_core + 273.15
        q_entropic = - current * t_kelvin * self.du_dt
        
        q_total = q_joule + q_entropic
        return max(0.0, q_total)

    def step(self, current: float, cooling_u: float, dt: float = 1.0, soc: float = 0.5, cycle: int = 1):
        """
        Advance thermal state by dt seconds.
        cooling_u: actuator control signal in [0.0, 1.0]
        Returns: (t_core, t_surf, q_gen, q_cool, p_pump)
        """
        u = float(np.clip(cooling_u, 0.0, 1.0))
        
        # Compute heat generation
        q_gen = self.compute_heat_generation(current, soc=soc, cycle=cycle)
        
        # Heat transfer rates
        q_cs = (self.t_core - self.t_surf) / self.r_cs
        q_amb = (self.t_surf - self.t_amb) / self.r_amb
        
        # Active cooling rate: proportional to control action u and temperature delta
        q_cool = u * self.k_cool_max * max(0.0, self.t_surf - self.t_coolant)
        
        # Differential equations
        d_t_core = (q_gen - q_cs) / self.c_core
        d_t_surf = (q_cs - q_amb - q_cool) / self.c_surf
        
        # Update states (Forward Euler)
        self.t_core += d_t_core * dt
        self.t_surf += d_t_surf * dt
        
        # Actuator power consumption (Affinity Law: P ~ u^3)
        p_pump = self.p_pump_max * (u ** 3)
        
        return {
            "t_core": self.t_core,
            "t_surf": self.t_surf,
            "q_gen": q_gen,
            "q_cool": q_cool,
            "q_amb": q_amb,
            "p_pump": p_pump,
            "cooling_u": u
        }
