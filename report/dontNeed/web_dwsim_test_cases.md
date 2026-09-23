# Web DWSIM Export Test Cases

This note provides two stable Web-console prompts for checking whether ThermoAgent can generate DWSIM `.dwxmz` files from the chat interface.

## Test Case 1: Binary VLE, 2-Propanol / Water

Paste this prompt into the Web chat box:

```text
请导出 2-丙醇-水 二元 VLE 精馏 DWSIM 文件，进料 2-丙醇 0.3，水 0.7，压力 101.325 kPa，进料流量 1 mol/s，塔顶 2-丙醇纯度 0.95。
```

Expected route:

```text
is_binary_vle_dwsim_request -> run_binary_distillation -> export_dwsim_binary_column
```

Expected Web behavior:

- The response payload contains `distillation.status = "ready"`.
- The response payload contains a non-null `distillation.dwsim_file_uri`.
- The browser auto-downloads a file named like `binary-<timestamp>-<id>.dwxmz`.

Verified local output:

```text
status = ready
dwsim_file_uri = /api/export/extractive/binary-20260916-221138-0e17f4b7.dwxmz
file = data/exports/extractive/binary-20260916-221138-0e17f4b7.dwxmz
```

Scientific meaning:

- The case is a binary VLE distillation design.
- Feed composition is ordered as `isopropanol / water = 0.3 / 0.7`.
- The DWSIM file contains a binary distillation column generated from the shortcut design values.

## Test Case 2: Binary LLE, Water / 1-Butanol

Paste this prompt into the Web chat box:

```text
请导出 水 / 正丁醇 二元 LLE DWSIM 文件，液液平衡，进料水 0.3，正丁醇 0.7，温度 298.15 K，压力 101.325 kPa，进料流量 1 mol/s。
```

Expected route:

```text
is_binary_lle_dwsim_request -> run_binary_lle_export -> export_dwsim_binary_lle_flowsheet
```

Expected Web behavior:

- The response payload contains `lle_extraction.status = "ready"`.
- The response payload contains a non-null `lle_extraction.dwsim_file_uri`.
- The browser auto-downloads a file named like `binary-lle-<timestamp>-<id>.dwxmz`.

Verified local output:

```text
status = ready
dwsim_file_uri = /api/export/extractive/binary-lle-20260916-221144-10211ce3.dwxmz
file = data/exports/extractive/binary-lle-20260916-221144-10211ce3.dwxmz
```

Scientific meaning:

- The case is a binary liquid-liquid equilibrium export, not a distillation column.
- Feed composition is ordered as `Water / 1-Butanol = 0.3 / 0.7`.
- The DWSIM file contains the topology `Feed -> Vessel_LLE -> Vapor / Light_Liquid / Heavy_Liquid`.
- LLE phase splitting is solved by DWSIM with the NRTL property package; ThermoAgent does not invent LLE equilibrium numbers.

## Notes

- Both cases require the prompt to explicitly include `DWSIM` or `导出`; otherwise the Web chat may return only a calculation/design response and not a downloadable `.dwxmz`.
- The LLE case must explicitly include `LLE` or `液液平衡`; otherwise it can be mistaken for a VLE or distillation request.
- For the water / 1-butanol case, write the feed composition in the same order as the normalized system name: `水 0.3，正丁醇 0.7`.
