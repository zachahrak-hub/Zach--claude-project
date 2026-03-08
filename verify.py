#!/usr/bin/env python3
"""Verify the GRC tab rewrite."""

filepath = '/Users/zach.ahrak/web-agent/templates/index.html'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.splitlines()

# Count ctrl-row occurrences
ctrl_rows = content.count('class="ctrl-row"')
print(f"ctrl-row count: {ctrl_rows} (expected 57)")
assert ctrl_rows == 57, f"FAIL: got {ctrl_rows} ctrl-rows"

# Find tab-grc boundaries
tab_grc_lines = [(i+1, l) for i, l in enumerate(lines) if 'tab-grc' in l]
print("tab-grc boundary lines:")
for lineno, text in tab_grc_lines:
    print(f"  line {lineno}: {text.strip()}")

# Verify opening tag exists
assert any('id="tab-grc"' in l for _, l in tab_grc_lines), "FAIL: opening tag not found"
assert any('/tab-grc' in l for _, l in tab_grc_lines), "FAIL: closing comment not found"

# Verify KPI IDs
for kpi_id in ['kpi-total2','kpi-passing2','kpi-failing2','kpi-inprog2','kpi-notstarted2']:
    assert kpi_id in content, f"FAIL: missing {kpi_id}"
print("KPI IDs: OK")

# Verify progress bar IDs
for pb_id in ['pb-passing2','pb-inprog2','pb-failing2','pb-notstarted2']:
    assert pb_id in content, f"FAIL: missing {pb_id}"
print("Progress bar IDs: OK")

# Verify JS functions present
for fn in ['saveStatus2','updateGRCKPIs','filterGRC2','searchGRC2','openJira2','syncJira2','toggleAgentRow']:
    assert fn in content, f"FAIL: missing JS function {fn}"
print("JS functions: OK")

# Verify all 12 sidebar categories
for cat in ['all','cc1','cc2','cc3','cc4','cc5','cc6','cc7','cc8','cc9','c1','a1']:
    assert f'data-cat="{cat}"' in content, f"FAIL: missing sidebar cat {cat}"
print("Sidebar categories: OK")

# Count data-id attributes (should be 57 ctrl-rows + 12 sidebar items = control rows only have data-id on ctrl-row divs)
data_ids_in_ctrl = [l for l in lines if 'class="ctrl-row"' in l and 'data-id=' in l]
print(f"ctrl-rows with data-id: {len(data_ids_in_ctrl)} (expected 57)")
assert len(data_ids_in_ctrl) == 57

# Verify CSS light theme override
assert '#f4f5f7' in content, "FAIL: missing light theme bg color"
assert '#0f172a' in content, "FAIL: missing header dark bg"
print("Light theme CSS: OK")

# Verify specific control IDs
for ctrl_id in ['1','2','3','5','621','7','8','10','13','9','11','12','1538','16','17','18',
                '20','21','22','47','26','23','24','25','29','30','31','32','33','34','35',
                '36','37','39','41','43','19','27','28','42','44','45','46','49','50','51',
                '53','54','525354','40','55','56','48','57','58','59','60']:
    assert f'data-id="{ctrl_id}"' in content, f"FAIL: missing control data-id={ctrl_id}"
print("All 57 control IDs: OK")

print("\nALL CHECKS PASSED.")
