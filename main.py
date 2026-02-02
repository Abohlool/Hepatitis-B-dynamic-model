
#~ Imports-------------------------------------------------------------------------------------------------------------
from plot import plot_deterministic, plot_controlled, plot_comparison, plot_controls, plot_R0
import customtkinter as ctk
from tkinter.messagebox import showerror

#~ GUI-Initialization--------------------------------------------------------------------------------------------------

app = ctk.CTk()
app.geometry("1250x925")
app.title("Dynamics of HBV")

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("./rime.json")


#~ GUI-----------------------------------------------------------------------------------------------------------------

header = ctk.CTkFrame(app, height=40)
header.pack(fill="x")
ctk.CTkLabel(header, text="Modeling the dynamics of HBV with Optimal Control", font=("Serif", 25)).pack(pady=10)

body = ctk.CTkFrame(app)
body.pack(fill="both", expand=True, padx=25, pady=25)

left_frame = ctk.CTkFrame(body)
left_frame.pack(side="left", fill="both", expand=True, padx=20)

ctk.CTkLabel(left_frame, text="Model Parameters:", font=("Serif", 20)).pack(pady=10)

#~ Inputs--------------------------------------------------------------------------------------------------------------

# Lambda, eta, mu_0, nu, alpha, gamma, gamma_1, beta, gamma_2, mu_1
parameter_labels = [(923, "Birth rate"), (951, "Maternally infected"), ((956, 0), "Natural Death"), (957, "Vaccination rate"), 
            (945, "Contact rate"), (947, "Reduced transmission rate"), ((947, 1), "Acute recovery rate"), (946, "Acute to chronic"), 
            ((947, 2), "Chronic recovery rate"), ((956, 1), "Infected death")]
parameters = list()
for i, s in parameter_labels:
    row = ctk.CTkFrame(left_frame)
    row.pack(fill="x", pady=4)
    
    if type(i) is tuple:
        char = f"{chr(i[0])}{i[1]}"
    else:
        char = chr(i)
    
    ctk.CTkLabel(row, text=f"{char} ({s}): ", width=180).pack(side="left", padx=10, anchor="e")
    entry = ctk.CTkEntry(row)
    entry.pack(side="right", fill="x", expand=True, padx=10)
    parameters.append(entry)


weights_labels = ["Acute contact weight", "Chronic contact weight", "Vaccination cost", "Treatment cost"]
weights = list()
for i in range(len(weights_labels)):
    row = ctk.CTkFrame(left_frame)
    row.pack(fill="x", pady=4)
    
    ctk.CTkLabel(row, text=f"w{i+1} ({weights_labels[i]}): ", width=180).pack(side="left", padx=10)
    
    entry = ctk.CTkEntry(row)
    entry.pack(side="left", fill="x", expand=True, padx=10)
    weights.append(entry)


initial_labels = ["S0", "A0", "B0", "R0"]
initials = list()
for population in initial_labels:
    row = ctk.CTkFrame(left_frame)
    row.pack(fill="x", pady=4)
    
    ctk.CTkLabel(row, text=f"{population}: ", width=180).pack(side="left", padx=10)
    slider = ctk.CTkSlider(row, from_=0, to=100, number_of_steps=20, width=400)
    slider.pack(padx=5)
    initials.append(slider)



times = list()

row = ctk.CTkFrame(left_frame)
row.pack(fill="x", pady=4)
ctk.CTkLabel(row, text=f"tf (Final time): ", width=180).pack(side="left", padx=10)
tf = ctk.CTkEntry(row)
tf.pack(side="left", fill="x", expand=True, padx=10)
times.append(tf)

row = ctk.CTkFrame(left_frame)
row.pack(fill="x", pady=4)
ctk.CTkLabel(row, text="dt (rk4 delta t): ", width=180).pack(side="left", padx=10)
dt = ctk.DoubleVar(value=0.05)
ctk.CTkComboBox(row, variable=dt, values=["0.01", "0.05", "0.1", "0.25", "0.5"]).pack(side="left", padx=10, expand=True)
times.append(dt)


right_frame = ctk.CTkFrame(body)
right_frame.pack(side="left", fill="both", expand=True, padx=20)
ctk.CTkLabel(right_frame, text="Models:", font=("Serif", 20)).pack(pady=(10, 20))


#$ -------------------------------------Getters-------------------------------------

def get_initials():
    Y0 = list()
    for ent in initials:
        Y0.append(int(ent.get()))
    return Y0


def get_parameters():
    try:
        Theta = list()
        for parameter in parameters:
            Theta.append(float(parameter.get().strip()))
        for param in Theta:
            if not 0 <= param <= 1:
                raise
    except:
        showerror("Error", "Invalid Model Parameter", detail="Enter a number between 0 and 1")
        raise
    else:
        return Theta


def get_weights():
    try:
        ws = list()
        for weight in weights:
            ws.append(float(weight.get().strip()))
        for w in ws:
            if not 0 <= w <= 1:
                raise
    except:
        showerror("Error", "Invalid Model Weights", detail="Enter a number between 0 and 1")
        raise
    else:
        return ws


def get_times():
    try:
        ts = list()
        for t in times:
            ts.append(float(t.get()))
        if not 0 <= ts[0]:
            raise
        if ts[1] <= 0:
            raise
    except:
        showerror("Error", "Invalid times", detail="Enter a number greater than 0")
        raise
    else:
        return ts


#$ -------------------------------------Button-Methods-------------------------------------

def deterministic():
    try:
        Y0 = get_initials()
        Theta = get_parameters()
        tf, dt = get_times()
    except:
        pass
    else:
        plot_deterministic(Y0, Theta, 0, tf, dt)


def controlled():
    try:
        Y0 = get_initials()
        Theta = get_parameters()
        weights = get_weights()
        tf, dt = get_times()
    except:
        pass
    else:
        plot_controlled(Y0, Theta, weights, 0, tf, dt)


def comparison():
    try:
        Y0 = get_initials()
        Theta = get_parameters()
        weights = get_weights()
        tf, dt = get_times()
    except:
        pass
    else:
        plot_comparison(Y0, Theta, weights, 0, tf, dt)


def controls():
    try:
        Y0 = get_initials()
        Theta = get_parameters()
        weights = get_weights()
        tf, dt = get_times()
    except:
        pass
    else:
        plot_controls(Y0, Theta, weights, 0, tf, dt)


def R0():
    try:
        Theta = get_parameters()
    except:
        pass
    else:
        plot_R0(Theta)


#~ Buttons-------------------------------------------------------------------------------------------------------------

deterministic_button = ctk.CTkButton(right_frame, text="Plot Deterministic Model", font=("Serif", 17), command=deterministic)
deterministic_button.pack(pady=5, padx=10, fill="x")

controlled_button = ctk.CTkButton(right_frame, text="Plot Controlled Model", font=("Serif", 17), command=controlled)
controlled_button.pack(pady=5, padx=10, fill="x")

comparison_button = ctk.CTkButton(right_frame, text="Plot Deterministic vs. Controlled", font=("Serif", 17), command=comparison)
comparison_button.pack(pady=5, padx=10, fill="x")

control_button = ctk.CTkButton(right_frame, text="Plot Control Functions", font=("Serif", 17), command=controls)
control_button.pack(pady=5, padx=10, fill="x")

R0_button = ctk.CTkButton(right_frame, text="Plot Reproductive Number over controls", font=("Serif", 17), command=R0)
R0_button.pack(pady=5, padx=10, fill="x")


app.mainloop()
