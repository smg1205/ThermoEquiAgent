# Guide to Using Extractive Distillation DWSIM Files

This document explains what is already present in the extractive distillation column `.dwxmz` file exported by
ThermoEqui-Agent, what is still missing, and how to complete the final step in the DWSIM graphical interface so
that the column can be calculated.

## 1. What the exported file already contains

The parts of the exported `Extractive Distillation Column` file that are already in place:

| Item | Already set |
|------|--------|
| Column type | rigorous column `DistillationColumn` (rigorous column) |
| Components | ethanol, water, entrainer (DWSIM canonical names) |
| Property package | the property package from the design (e.g. NRTL) |
| Feed stream | `Ethanol Water Feed` → column feed port (feed stage) |
| Entrainer stream | `Entrainer` → the column's second feed port |
| Distillate product | `Ethanol Product` (distillate) |
| Bottoms product | `Water Entrainer Bottoms` (bottoms) |
| Number of stages | as designed (e.g. 12) |
| Reflux ratio | as designed (e.g. 2.772) |

In other words, **the material connections and the basic column parameters are already configured**, and opening
DWSIM shows a structurally complete flowsheet.

## 2. Why one more step is needed (limitation note)

Besides the material connections, DWSIM's rigorous column `Validate()` also requires the **solver specifications
inside the column** to be complete, for example:

- the **specification** of the condenser (total condenser) (specifying the distillate via reflux ratio / distillate flow rate / purity, etc.)
- the **specification** of the reboiler (specifying the bottoms via boilup ratio / bottoms flow rate / heat duty, etc.)

These **internal column specifications** have no stable, reliable setting interface in the DWSIM Automation API;
they are content that the user configures in the column parameter interface after opening the file. The exported
file can therefore be generated with correct connections, but **before the first calculation a condenser
specification and a reboiler specification must be added in DWSIM**. Please follow the steps in section 3.

> This is the capability boundary of DWSIM Automation, not a bug in this project's code. The downstream exporter
> (DWSIM) requires a complete solver structure, and that structure can only be fully specified in the GUI.

## 3. Making the column calculable in DWSIM (steps)

Assume you have already opened the exported `.dwxmz` file.

### Step A: Confirm the flowsheet connections are complete

1. On the DWSIM canvas, look at the `Extractive Distillation Column` column.
2. You should see 4 material connection lines:
   - `Ethanol Water Feed` → column (feed)
   - `Entrainer` → column (entrainer, the second feed port)
   - column → `Ethanol Product`
   - column → `Water Entrainer Bottoms`
3. If any is missing, add it with the connection tool in the toolbar (pencil/wire icon).

### Step B: Add a specification to the condenser

1. Double-click the column to open the column configuration window (Column Configuration).
2. Find the **Condenser** page.
3. **Condenser Type** should remain **Total Condenser**.
4. In the Specification area, choose a specification method and enter a value, for example:
   - **Reflux Ratio = 2.772** (using the design reflux ratio), or
   - **Distillate Flow Rate** (specifying the distillate production rate).
5. Apply.

### Step C: Add a specification to the reboiler

1. In the same column configuration window, find the **Reboiler** page.
2. In the Specification area, choose a specification method and enter a value, for example:
   - **Boilup Ratio** (start with an initial estimate close to 1, then fine-tune), or
   - **Bottoms Flow Rate** (specifying the bottoms production rate), or
   - directly give an initial estimate of the **Reboiler Duty**.
3. Apply.

### Step D: Attach energy streams (optional but recommended)

Calculating a rigorous column requires the heat duty of the condenser and the reboiler:

1. In the object panel, find **Energy Stream** (in some versions it is called Utility/Heat).
2. Place an energy stream object near the **overhead condenser**, and use the connection tool to connect from the **condenser energy port** to it.
3. Then place an energy stream object near the **bottoms reboiler**, and connect from the **reboiler energy port** to it.
4. (If you specify the column with the ratio/flow-rate specifications from steps B/C, the energy streams can have their duty calculated automatically; if the energy ports are not visible on the canvas,
   they can also be omitted, and DWSIM uses a default duty estimate.)

### Step E: Run the calculation

1. Click DWSIM's **Calculate / Run / Solve** (or F5).
2. If it still reports missing connections, go back to steps A/B/C and check whether any port is unpaired.

## 4. Expected results

After completing the condenser/reboiler specifications, the rigorous column should pass `Validate()` and begin
iterative solving, yielding the distillate and bottoms compositions, temperatures, heat duties of each module, and
other results. The design parameters (number of stages, reflux ratio) have already been written into the file and
can serve as initial values.

## 5. Frequently asked questions

- **It reports `One or more of the stream connections to the column is missing`**
  This is usually because the condenser/reboiler specifications or the energy connections have not been completed. Check per steps B/C/D in section 3.
- **The energy port / energy stream is not visible**
  Zoom out on the canvas and look at the edge of the column icon; or search the object panel for "Energy" / "Utility".
- **I want to use a shortcut column (ShortcutColumn) for a quick estimate**
  A shortcut column can directly compute the minimum stages/reflux, but it supports only a single feed (it cannot express the
  real extractive structure of "feed + entrainer on different stages"). This project's exports use a rigorous column to preserve the correctness of the extractive structure.

## 6. DWSIM version confirmation

This guide targets the DWSIM desktop version. The wording of menus may differ between minor versions (for example,
the location of "Reboiler Duty" differs), but the approach is the same: **add the condenser specification + add the
reboiler specification + attach energy streams + calculate**.
