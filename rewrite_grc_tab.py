#!/usr/bin/env python3
"""Rewrites the GRC-SOC2 tab in templates/index.html between lines 934 and 2774 (inclusive)."""

import re

# ── helpers ──────────────────────────────────────────────────────────────────

def badge_class(badge):
    return {'auto': 'badge-auto', 'maybe': 'badge-maybe', 'manual': 'badge-manual-2'}[badge]

def badge_label(badge):
    return {'auto': '⚡ Auto', 'maybe': '~ Maybe Auto', 'manual': '✎ Manual'}[badge]

def ctrl_row(id_, ccid, crlg, title, desc, badge, tools, task, cat):
    sid = str(id_)
    safe_title = title.replace("'", "\\'")
    tool_html = ''
    if tools:
        spans = ''.join(f'<span class="ctrl-tool">{t}</span>' for t in tools)
        tool_html = f'<div class="ctrl-tools">{spans}</div>'

    agent_html = ''
    expand_btn = ''
    if task:
        safe_task  = task.replace("'", "\\'").replace('\n', ' ')
        agent_html = f'''
    <div class="ctrl-agent-wrap" id="agent-wrap-{sid}">
      <div class="ctrl-agent-row2">
        <button class="btn-run-agent2" onclick="runGRC('{sid}','{crlg.replace('#','_')}','{safe_title}','{safe_task}')">&#9654; Run Agent</button>
        <div class="grc-status2" id="status-{sid}"><span class="status-dot" id="dot-{sid}"></span><span id="status-text-{sid}">Not run</span></div>
      </div>
      <div class="grc-result2" id="result-{sid}"></div>
    </div>'''
        expand_btn = f'<button class="ctrl-expand-btn" onclick="toggleAgentRow(\'{sid}\')" title="Toggle agent">&#9889;</button>'

    return f'''              <div class="ctrl-row" data-id="{sid}" data-cat="{cat}">
                <span class="ctrl-status-sq not_started" id="sq-{sid}"></span>
                <div class="ctrl-body">
                  <div class="ctrl-meta">
                    <span class="ctrl-ccid">{ccid}</span>
                    <span class="ctrl-crlg-badge">{crlg}</span>
                    {tool_html}
                    <span class="ctrl-auto-badge {badge_class(badge)}">{badge_label(badge)}</span>
                  </div>
                  <div class="ctrl-title">{title}</div>
                  <div class="ctrl-desc">{desc}</div>{agent_html}
                </div>
                <div class="ctrl-right">
                  <select class="status-sel2 not_started" onchange="saveStatus2('{sid}', this.value, this)">
                    <option value="not_started">Not Started</option>
                    <option value="in_progress">In Progress</option>
                    <option value="passing">Passing</option>
                    <option value="failing">Failing</option>
                  </select>
                  <button class="btn-jira2" onclick="openJira2('{safe_title}','{ccid}')">+ Jira</button>
                  {expand_btn}
                </div>
              </div>'''

# ── control data ─────────────────────────────────────────────────────────────

controls = [
    # CC1
    dict(id_=1,   ccid='CC1.1', crlg='CRLG#1',  cat='cc1', title='Board Independence',          desc='Board members and the management members list confirming independent board formation.',                                                                                                                 badge='manual', tools=[], task=None),
    dict(id_=2,   ccid='CC1.2', crlg='CRLG#2',  cat='cc1', title='Board Policy Direction',       desc='Sample of Board meeting minutes regarding setting policies and ethical behavior.',                                                                                                                    badge='manual', tools=[], task=None),
    dict(id_=3,   ccid='CC1.3', crlg='CRLG#3',  cat='cc1', title='Organizational Structure',     desc='An updated organization chart confirming structure is reviewed and updated at least annually.',                                                                                                      badge='manual', tools=[], task=None),
    dict(id_=5,   ccid='CC1.4', crlg='CRLG#5',  cat='cc1', title='New Employee Onboarding',      desc='For a selected sample of new employees during the audit period, provide the training tracking table.',                                                                                               badge='manual', tools=[], task=None),
    dict(id_=621, ccid='CC1.5', crlg='CRLG#6',  cat='cc1', title='Policy Documents',             desc='All company policies and procedures (Employee Handbook, Information Security Policies, SDLC, Incident Response, Hardening). Screenshot of internal drive confirming policies are up to date.',        badge='auto',   tools=['fetch_google_drive_policies'], task='Download the latest security and compliance policies from the relevant Google Drive folder and verify they are updated (Employee Handbook, Information Security Policies, SDLC, Incident Response, Hardening). Confirm the policies are placed in the correct evidence folder. Report what was found and any gaps.'),
    dict(id_=7,   ccid='CC1.6', crlg='CRLG#7',  cat='cc1', title='Job Requirements & Hiring',   desc='For a sample of new employees during the audit period, provide documentation of the recruitment process (background checks, reference checks, interview minutes, etc.).',                             badge='manual', tools=[], task=None),
    dict(id_=8,   ccid='CC1.7', crlg='CRLG#8',  cat='cc1', title='Management Team Meetings',    desc='Sample of management meeting calendar invitations (5 weeks) and meeting agenda covering security, confidentiality and availability issues.',                                                          badge='manual', tools=[], task=None),
    dict(id_=10,  ccid='CC1.8', crlg='CRLG#10', cat='cc1', title='Active Employee List',         desc='A list of current active employees across all entities. BDO will provide a sample for annual performance review evidence.',                                                                           badge='auto',   tools=['HiBob'], task='Export the full list of active employees from HiBob as of today. Include: name, department, location, start date, and employment status. Flag any accounts that appear inactive but are still enabled.'),
    dict(id_=13,  ccid='CC1.9', crlg='CRLG#13', cat='cc1', title='Security Awareness Training', desc='Security awareness training course agenda and completion tracking for all employees.',                                                                                                                 badge='manual', tools=[], task=None),
    # CC2
    dict(id_=9,    ccid='CC2.1', crlg='CRLG#9',  cat='cc2', title='Whistleblower / Ethical Reports', desc='Whistleblower policy and evidence of ethical reporting mechanism communicated to all employees.',                                                                                              badge='manual', tools=[], task=None),
    dict(id_=11,   ccid='CC2.2', crlg='CRLG#11', cat='cc2', title='Confidentiality Changes',          desc='Confidentiality commitments change notification sent to all relevant internal and external parties.',                                                                                         badge='manual', tools=[], task=None),
    dict(id_=12,   ccid='CC2.3', crlg='CRLG#12', cat='cc2', title='Customer Responsibilities / SLA',  desc='SLA document and evidence that the SLA is published on the company website and communicated to customers.',                                                                                  badge='manual', tools=[], task=None),
    dict(id_=1538, ccid='CC2.4', crlg='CRLG#15', cat='cc2', title='Customer Population',              desc='A list of new customers during the audit period. BDO will select a sample to verify contract terms and onboarding.',                                                                         badge='maybe',  tools=['Salesforce'], task='Export the list of new customers added during the audit period from Salesforce. Include: company name, contract start date, ARR, and assigned CSM. BDO will sample from this list.'),
    dict(id_=16,   ccid='CC2.5', crlg='CRLG#16', cat='cc2', title='Incident Response Communication',  desc='Evidence that incident response policies and procedures are communicated to relevant internal and external users on Coralogix website.',                                                      badge='manual', tools=[], task=None),
    dict(id_=17,   ccid='CC2.6', crlg='CRLG#17', cat='cc2', title='Customer Updates / Status Page',   desc="Evidence that new features and modifications are communicated to customers via Coralogix's status page and emails.",                                                                          badge='manual', tools=[], task=None),
    dict(id_=18,   ccid='CC2.7', crlg='CRLG#18', cat='cc2', title='How-To / FAQ',                     desc='Screenshot of the company\'s FAQ and support manual (website) and screenshot showing the end user can ask questions from the Coralogix platform.',                                           badge='manual', tools=[], task=None),
    # CC3
    dict(id_=20, ccid='CC3.1', crlg='CRLG#20', cat='cc3', title='Asset Inventory Export',        desc="A master list of the entity's system component and technology assets, accounting for additions and removals during the audit period.",                                                             badge='auto',   tools=['AWS', 'Platform'], task='Export the full asset inventory as of the audit date. List all infrastructure assets grouped by environment (prod/stag/test), with asset type, owner, and data classification. Flag any untagged or unclassified assets.'),
    dict(id_=21, ccid='CC3.2', crlg='CRLG#21', cat='cc3', title='Risk Management Process',      desc="The company's Information Security Risk Management Policy confirming a formal risk management process and risk mitigation strategy is in place.",                                                  badge='manual', tools=[], task=None),
    dict(id_=22, ccid='CC3.3', crlg='CRLG#22', cat='cc3', title='Annual Risk Assessment (ERA)', desc="Company's approved annual Enterprise Risk Assessment (ERA), reviewed and approved by senior management.",                                                                                          badge='manual', tools=[], task=None),
    dict(id_=47, ccid='CC3.4', crlg='CRLG#47', cat='cc3', title='Risk-Based Change Requests',   desc="Company's approved annual ERA and evidence that change requests were created based on identified needs from the risk assessment.",                                                                 badge='manual', tools=[], task=None),
    # CC4
    dict(id_=26, ccid='CC4.1', crlg='CRLG#26', cat='cc4', title='Internal Audit / Compliance Monitoring', desc='Internal compliance monitoring report confirming controls are evaluated on an ongoing basis. CSPM scan results.', badge='auto', tools=['Coralogix CSPM'], task='Pull the latest compliance monitoring report from Coralogix CSPM. Summarize: number of controls monitored, pass/fail ratio, critical gaps, and last scan date.'),
    # CC5
    dict(id_=23, ccid='CC5.1', crlg='CRLG#23', cat='cc5', title='DR Test / Business Recovery',        desc='Business recovery plan and last DR test results. Recovery plans including restoration of backups are tested annually.',                                                                       badge='manual', tools=[], task=None),
    dict(id_=24, ccid='CC5.2', crlg='CRLG#24', cat='cc5', title='Penetration Test / SAST Report',     desc='Internal/external vulnerability tests including results (penetration test + SAST vulnerability tests). Annual pen test by independent third party.',                                           badge='auto',   tools=['Orca Security'], task='From Orca Security, export all SAST vulnerabilities identified on source code during the audit period. Report totals by severity (Critical/High/Medium/Low), remediation rate, and any unresolved critical items.'),
    dict(id_=25, ccid='CC5.3', crlg='CRLG#25', cat='cc5', title='Remediation Plans',                  desc='Remediation plans established based upon severity level of detected vulnerabilities. Evidence of tracking and resolution.',                                                                   badge='auto',   tools=['CloudWatch', 'Coralogix CSPM', 'Okta'], task='From CloudWatch and Coralogix CSPM, list all open findings with severity >= High. For each finding: show status, assigned owner, and target remediation date. Flag overdue items.'),
    # CC6
    dict(id_=29,  ccid='CC6.1',  crlg='CRLG#29', cat='cc6', title='Physical Security (Data Center)',      desc='Physical access controls for the data center (colocation or cloud). Evidence of restricted physical access to servers.',                                                                  badge='manual', tools=[], task=None),
    dict(id_=30,  ccid='CC6.2',  crlg='CRLG#30', cat='cc6', title='AWS Prod Access (IAM + SSO)',           desc='AWS user list and permissions by role. SSO configuration enabled. All production access via Okta SSO/SAML.',                                                                            badge='auto',   tools=['Okta', 'AWS IAM', 'SSO/SAML'], task='Export all AWS IAM users and roles with production access. Cross-reference with Okta SSO assignments. Confirm all access is via SSO/SAML. Flag any users with direct IAM access or overly permissive policies (Admin, *).'),
    dict(id_=31,  ccid='CC6.3',  crlg='CRLG#31', cat='cc6', title='SSO / Password Policy',               desc='Policies of SSO to company programs. Screenshot of password policy settings in Okta (MFA, session timeout, complexity).',                                                                badge='maybe',  tools=['Okta'], task='From Okta, export the current password policy and MFA enforcement settings. Confirm: minimum password length >= 12, MFA required for all users, session timeout <= 8 hours. Flag any gaps.'),
    dict(id_=32,  ccid='CC6.4',  crlg='CRLG#32', cat='cc6', title='Admin Access List',                   desc='A list of Okta admins and a list of AWS admins. Requires Okta and AWS admin access to export.',                                                                                          badge='maybe',  tools=['Okta', 'AWS'], task='Export the list of Okta admins and AWS account admins. For each: confirm business justification and last access date. Flag any admins who have not logged in for 90+ days.'),
    dict(id_=33,  ccid='CC6.5',  crlg='CRLG#33', cat='cc6', title='Prod Environment & DB Access',        desc='List of users with access to the database and production environment, with their access roles and justification.',                                                                         badge='auto',   tools=['Okta', 'AWS IAM'], task='List all users with direct database and production environment access. Include role, access method, and last login. Confirm access is restricted to authorized personnel only.'),
    dict(id_=34,  ccid='CC6.6',  crlg='CRLG#34', cat='cc6', title='GitHub / Source Control Access',      desc='List of users with access to GitHub. SSO enabled for all GitHub access via Okta.',                                                                                                        badge='auto',   tools=['Okta', 'GitHub'], task='Export all GitHub organization members and their repository access levels. Confirm SSO is enforced. Flag external collaborators and users with admin access.'),
    dict(id_=35,  ccid='CC6.7',  crlg='CRLG#35', cat='cc6', title='Quarterly Access Review',             desc='User permissions review approved by relevant managers for Q1-Q4 of the audit period.',                                                                                                   badge='auto', tools=['fetch_hibob_employees'], task='Export current employee roster from HiBob and review access permissions for accuracy. Verify all active employees have appropriate access and inactive employees are deprovisioned.'),
    dict(id_=36,  ccid='CC6.8',  crlg='CRLG#36', cat='cc6', title='New Hire Population',                 desc='For a sample of new employees during the audit period, provide evidence of access provisioning (Okta, GitHub, AWS).',                                                                    badge='auto',   tools=['fetch_hibob_employees'], task='From HiBob, export all employees hired during the audit period. Confirm each new hire completed onboarding (background check, security training, system access provisioned via Okta). Flag any gaps.'),
    dict(id_=37,  ccid='CC6.9',  crlg='CRLG#37', cat='cc6', title='Offboarding Population',              desc='List of terminated employees during the audit period. Confirm accounts deactivated within 24 hours.',                                                                                    badge='auto',   tools=['HiBob'], task='From HiBob, export all terminated employees during the audit period. For each: confirm Okta account deactivated within 24 hours, AWS access removed, and laptop returned. Flag any gaps.'),
    dict(id_=39,  ccid='CC6.10', crlg='CRLG#39', cat='cc6', title='Remote Access / VPN (Teleport)',      desc='Teleport users list for production environment access. Evidence of SSO authentication via Okta for all Teleport sessions.',                                                               badge='auto',   tools=['Teleport', 'Okta'], task='Export all active Teleport users with production environment access. Confirm access is SSO-authenticated via Okta. List users with their assigned roles and last session date. Flag unused accounts (no session in 60+ days).'),
    dict(id_=41,  ccid='CC6.11', crlg='CRLG#41', cat='cc6', title='Firewall / Prod Security Groups',     desc='Screenshot of FW rules, last update date, and users with permission to modify security groups in AWS.',                                                                                   badge='auto',   tools=['AWS Security Groups'], task='Export all AWS security group rules for production environments. Flag any rules allowing 0.0.0.0/0 (open to internet) on non-standard ports. Report total rules, flagged rules, and last modification date.'),
    dict(id_=43,  ccid='CC6.12', crlg='CRLG#43', cat='cc6', title='SFTP / Encryption in Transit',       desc='Evidence of SSL/SFTP/HTTPS protocols in use for all defined data flows, especially customer-facing APIs and file transfers.',                                                             badge='manual', tools=[], task=None),
    # CC7
    dict(id_=19,  ccid='CC7.1', crlg='CRLG#19', cat='cc7', title='CS Incidents Report',            desc='Log tracker report from Intercom for Customer Support service tickets during the audit period. BDO will sample tickets to verify SLA compliance.',                                             badge='auto',   tools=['Intercom'], task='Generate a customer support incident report from Intercom filtered to the audit period. List total tickets, resolved vs open, average resolution time, SLA compliance, and any breaches. Note any access limitations.'),
    dict(id_=27,  ccid='CC7.2', crlg='CRLG#27', cat='cc7', title='Incident Population',             desc='List of security incidents for the audit period from PagerDuty. BDO will sample incidents to verify resolution.',                                                                            badge='auto',   tools=['PagerDuty', 'Jira'], task='Pull all security incidents from the audit period from PagerDuty. List each with: date, severity, systems affected, root cause, and resolution date. Confirm all P1/P2 incidents have a post-mortem linked.'),
    dict(id_=28,  ccid='CC7.3', crlg='CRLG#28', cat='cc7', title='IT Security Meetings',            desc='Sample of security and IT meeting calendar invitations (5 weeks), meeting agenda, and IT security tickets for action items.',                                                                badge='manual', tools=[], task=None),
    dict(id_=42,  ccid='CC7.4', crlg='CRLG#42', cat='cc7', title='Endpoint Antivirus Report',       desc='Antivirus installed on all production systems and employee laptops. AV logs report for last month verifying daily scans and updated virus definitions.',                                      badge='auto',   tools=['JAMF'], task='From JAMF, generate a report of all employee-managed laptops. Confirm antivirus is installed and active on each device. Count compliant vs non-compliant endpoints and list any gaps.'),
    dict(id_=44,  ccid='CC7.5', crlg='CRLG#44', cat='cc7', title='Security Events Escalation',      desc='Root cause analysis for any information security incidents during the audit period. Summary report from Cloudflare.',                                                                         badge='manual', tools=[], task=None),
    dict(id_=45,  ccid='CC7.6', crlg='CRLG#45', cat='cc7', title='Root Cause Analysis',             desc='Evidence of change requests created following high severity incidents, including root cause analysis reviewed by operations management.',                                                    badge='manual', tools=[], task=None),
    # CC8
    dict(id_=46,     ccid='CC8.1', crlg='CRLG#46', cat='cc8', title='Change Management Policy',         desc='Evidence of change management policy and procedure. GitHub pull request process as the formal change management workflow.',                                                               badge='auto',   tools=['GitHub'], task='From GitHub, export all pull requests merged to main/master during the audit period. Confirm each PR has: at least one approving reviewer, passing CI checks, and linked ticket. Flag any direct commits bypassing PR review.'),
    dict(id_=49,     ccid='CC8.2', crlg='CRLG#49', cat='cc8', title='Change Documentation (Jira)',      desc='Evidence for planning/sprint meeting (5 weeks) and screenshot of Jira backlog showing change requests.',                                                                                badge='manual', tools=[], task=None),
    dict(id_=50,     ccid='CC8.3', crlg='CRLG#50', cat='cc8', title='Automated Tests (Cypress)',        desc='Evidence of automatic tests running regularly (status from CI/CD pipeline showing test results).',                                                                                       badge='manual', tools=[], task=None),
    dict(id_=51,     ccid='CC8.4', crlg='CRLG#51', cat='cc8', title='Branch Protection Rules',          desc='GitHub branch protection rules for master/main on the two in-scope repositories.',                                                                                                      badge='auto',   tools=['GitHub'], task='Export branch protection rules for all main/master branches across the two in-scope GitHub repositories. Confirm: required reviewers >= 1, status checks required, force push disabled, and admin enforcement.'),
    dict(id_=53,     ccid='CC8.5', crlg='CRLG#53', cat='cc8', title='IaC Repo Access',                  desc='List of users with permissions to approve security, code, and infrastructure changes in GitHub.',                                                                                       badge='auto',   tools=['GitHub'], task='List all users with write/admin permissions to infrastructure-as-code repositories on GitHub. Confirm each user has business justification. Flag anyone outside the DevOps/Security team.'),
    dict(id_=54,     ccid='CC8.6', crlg='CRLG#54', cat='cc8', title='Deployment Notifications',         desc='Deployment notification evidence in Slack/Emails confirming successful deployments to production.',                                                                                     badge='manual', tools=[], task=None),
    dict(id_=525354, ccid='CC8.7', crlg='CRLG#52', cat='cc8', title='Change Platform Data (Code Review)', desc='List of changes deployed to production from GitHub during the audit period, with code review and CI/CD evidence.',                                                                  badge='auto',   tools=['GitHub', 'Terraform', 'CI/CD', 'AWS RDS', 'S3'], task='List all changes deployed to production from GitHub during the audit period. For each deployment: confirm code review, CI/CD pipeline pass, and post-deploy validation. Flag any deployments without approval or failed checks.'),
    # CC9
    dict(id_=40, ccid='CC9.1', crlg='CRLG#40', cat='cc9', title='Business Insurance',           desc="Evidence of business insurance coverage appropriate to the entity's risk profile.",                                                                                                              badge='manual', tools=[], task=None),
    dict(id_=55, ccid='CC9.2', crlg='CRLG#55', cat='cc9', title='Vendor Population',            desc='List of new vendors and related parties during the audit period. BDO will sample vendors for due diligence evidence.',                                                                          badge='auto',   tools=['Vendor Registry'], task='From the internal vendor registry, export all active vendors used during the audit period. Include: vendor name, service type, data access level, and last risk assessment date. Flag vendors with data access but no completed assessment.'),
    dict(id_=56, ccid='CC9.3', crlg='CRLG#56', cat='cc9', title='Vendor Compliance Assessment', desc='Vendor risk assessment for critical services and vendor due-diligence evidence (SOC 2 reports, security questionnaires).',                                                                      badge='manual', tools=[], task=None),
    # C1
    dict(id_=48, ccid='C1.1', crlg='CRLG#48', cat='c1', title='NDA / Confidentiality Agreements', desc='NDA and confidentiality agreements executed with employees, contractors, and relevant third parties.', badge='manual', tools=[], task=None),
    # A1
    dict(id_=57, ccid='A1.1', crlg='CRLG#57', cat='a1', title='Backup Configuration',    desc='AWS backup configuration for all production RDS and S3 resources. Confirm automated backups enabled with 30-day retention.',                                                                         badge='auto', tools=['AWS RDS', 'S3'],                               task='From AWS, export backup configurations for all production RDS instances and S3 buckets. Confirm: automated backups enabled, retention >= 30 days, backup tests performed. List any resources with missing or insufficient backups.'),
    dict(id_=58, ccid='A1.2', crlg='CRLG#58', cat='a1', title='Multi-AZ Replication',    desc='AWS monitor confirming availability zones are active. Infrastructure deployed across multiple AZs.',                                                                                                badge='auto', tools=['AWS RDS'],                                     task='From AWS RDS, export the configuration for all production database instances. Confirm Multi-AZ deployment is enabled. List all instances with their AZ configuration and failover status.'),
    dict(id_=59, ccid='A1.3', crlg='CRLG#59', cat='a1', title='Redundancy',              desc='Evidence that redundancy has been implemented for all critical infrastructure components.',                                                                                                          badge='auto', tools=['AWS'],                                         task='From AWS, confirm redundancy is implemented for all critical components (load balancers, compute, databases). Report: services with multi-AZ, auto-scaling groups, and health check configurations. Flag single points of failure.'),
    dict(id_=60, ccid='A1.4', crlg='CRLG#60', cat='a1', title='Monitoring & Alerts',     desc='Prometheus dashboard screenshot. Prometheus monitor alert configuration and on-call schedule.',                                                                                                    badge='auto', tools=['Prometheus', 'Grafana', 'CloudWatch'],          task='From CloudWatch and Prometheus/Grafana, export all active monitoring alerts for production. Confirm: uptime monitors active, latency and error rate alerts configured, on-call rotation set up. List any critical services without monitoring coverage.'),
]

assert len(controls) == 57, f"Expected 57 controls, got {len(controls)}"

# ── build all rows ────────────────────────────────────────────────────────────
rows_html = '\n'.join(ctrl_row(**c) for c in controls)

# ── assemble tab HTML ─────────────────────────────────────────────────────────
tab_html = r'''    <div class="tab-content" id="tab-grc">
      <style>
        #tab-grc {
          background: #f4f5f7 !important;
          min-height: 100vh;
          padding: 0 !important;
          color: #1e293b;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        .soc-wrap { display: flex; flex-direction: column; height: 100%; }

        /* HEADER */
        .soc-hdr { background: #0f172a; padding: 18px 32px; display: flex; align-items: center; justify-content: space-between; }
        .soc-hdr-title { font-size: 1.25rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 8px; }
        .soc-hdr-sub { font-size: .8rem; color: #94a3b8; margin-top: 2px; }
        .btn-sync-jira { background: #2563eb; color: #fff; border: none; border-radius: 6px; padding: 9px 18px; font-size: .85rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 6px; }
        .btn-sync-jira:hover { background: #1d4ed8; }

        /* KPI CARDS */
        .soc-kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: #e2e8f0; border-bottom: 1px solid #e2e8f0; }
        .soc-kpi { background: #fff; padding: 20px 24px; text-align: center; }
        .soc-kpi-num { font-size: 2rem; font-weight: 700; line-height: 1; margin-bottom: 6px; }
        .soc-kpi-label { font-size: .72rem; font-weight: 600; letter-spacing: .06em; color: #64748b; text-transform: uppercase; display: flex; align-items: center; justify-content: center; gap: 4px; }
        .kpi-total .soc-kpi-num { color: #1e293b; }
        .kpi-passing .soc-kpi-num { color: #16a34a; }
        .kpi-failing .soc-kpi-num { color: #dc2626; }
        .kpi-inprog .soc-kpi-num { color: #d97706; }
        .kpi-notstarted .soc-kpi-num { color: #94a3b8; }

        /* PROGRESS */
        .soc-progress-section { background: #fff; border-bottom: 1px solid #e2e8f0; padding: 16px 32px; }
        .soc-progress-title { font-size: .85rem; font-weight: 600; color: #1e293b; margin-bottom: 10px; }
        .soc-progress-bar { height: 8px; background: #e2e8f0; border-radius: 999px; overflow: hidden; display: flex; }
        .pb-seg { height: 100%; transition: width .4s; }
        .pb-passing { background: #16a34a; }
        .pb-inprog { background: #d97706; }
        .pb-failing { background: #dc2626; }
        .pb-notstarted { background: #e2e8f0; }
        .soc-progress-legend { display: flex; gap: 20px; margin-top: 8px; }
        .prog-legend-item { display: flex; align-items: center; gap: 5px; font-size: .75rem; color: #64748b; }
        .prog-dot { width: 8px; height: 8px; border-radius: 50%; }

        /* BODY */
        .soc-body { display: flex; flex: 1; min-height: 0; }

        /* SIDEBAR */
        .soc-sidebar { width: 220px; min-width: 220px; background: #fff; border-right: 1px solid #e2e8f0; padding: 12px 0; overflow-y: auto; }
        .soc-sb-item { padding: 10px 20px; cursor: pointer; border-left: 3px solid transparent; transition: all .15s; }
        .soc-sb-item:hover { background: #f8fafc; }
        .soc-sb-item.active { border-left-color: #2563eb; background: #eff6ff; }
        .soc-sb-cc { font-size: .75rem; font-weight: 700; color: #2563eb; margin-bottom: 1px; }
        .soc-sb-item.active .soc-sb-cc { color: #1d4ed8; }
        .soc-sb-name { font-size: .8rem; color: #475569; }
        .soc-sb-item.active .soc-sb-name { color: #1e40af; font-weight: 500; }

        /* PANEL */
        .soc-panel { flex: 1; overflow-y: auto; background: #f4f5f7; }
        .soc-search-wrap { padding: 16px 24px; background: #fff; border-bottom: 1px solid #e2e8f0; }
        .soc-search { width: 100%; padding: 9px 14px 9px 36px; border: 1px solid #e2e8f0; border-radius: 6px; font-size: .875rem; color: #1e293b; background: #f8fafc url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%2394a3b8' viewBox='0 0 16 16'%3E%3Cpath d='M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398l3.85 3.85a1 1 0 0 0 1.415-1.415l-3.868-3.833zm-5.242 1.406a5 5 0 1 1 0-10 5 5 0 0 1 0 10z'/%3E%3C/svg%3E") no-repeat 12px center; outline: none; box-sizing: border-box; }
        .soc-search:focus { border-color: #2563eb; background-color: #fff; }
        .soc-list-hdr { display: flex; align-items: baseline; justify-content: space-between; padding: 16px 24px 8px; }
        .soc-list-hdr-left h3 { font-size: 1.1rem; font-weight: 700; color: #1e293b; margin: 0 0 2px; }
        .soc-list-hdr-left p { font-size: .78rem; color: #64748b; margin: 0; }
        .soc-list-count { font-size: .78rem; color: #64748b; white-space: nowrap; }
        .soc-list { background: #fff; margin: 0 24px 24px; border-radius: 8px; border: 1px solid #e2e8f0; overflow: hidden; }

        /* CONTROL ROW */
        .ctrl-row { display: flex; align-items: flex-start; gap: 14px; padding: 14px 20px; border-bottom: 1px solid #f1f5f9; transition: background .1s; }
        .ctrl-row:last-child { border-bottom: none; }
        .ctrl-row:hover { background: #f8fafc; }
        .ctrl-status-sq { width: 16px; height: 16px; min-width: 16px; border-radius: 3px; margin-top: 2px; border: 1.5px solid #cbd5e1; background: transparent; cursor: pointer; }
        .ctrl-status-sq.passing { background: #16a34a; border-color: #16a34a; }
        .ctrl-status-sq.failing { background: #dc2626; border-color: #dc2626; }
        .ctrl-status-sq.in_progress { background: #d97706; border-color: #d97706; }
        .ctrl-status-sq.not_started { background: transparent; border-color: #cbd5e1; }
        .ctrl-body { flex: 1; min-width: 0; }
        .ctrl-meta { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; flex-wrap: wrap; }
        .ctrl-ccid { font-size: .72rem; font-weight: 700; color: #2563eb; }
        .ctrl-crlg-badge { font-size: .65rem; background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; border-radius: 4px; padding: 1px 6px; }
        .ctrl-auto-badge { font-size: .65rem; border-radius: 4px; padding: 1px 6px; }
        .badge-auto { background: #f0fdf4; color: #16a34a; border: 1px solid #bbf7d0; }
        .badge-maybe { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
        .badge-manual-2 { background: #f8fafc; color: #94a3b8; border: 1px solid #e2e8f0; }
        .ctrl-tools { display: flex; gap: 4px; flex-wrap: wrap; }
        .ctrl-tool { font-size: .65rem; background: #eff6ff; color: #3b82f6; border: 1px solid #bfdbfe; border-radius: 4px; padding: 1px 6px; }
        .ctrl-title { font-size: .9rem; font-weight: 600; color: #1e293b; margin-bottom: 2px; }
        .ctrl-desc { font-size: .8rem; color: #64748b; line-height: 1.5; }
        .ctrl-agent-wrap { margin-top: 8px; display: none; }
        .ctrl-agent-wrap.expanded { display: block; }
        .ctrl-agent-row2 { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
        .btn-run-agent2 { font-size: .75rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 5px; padding: 4px 12px; cursor: pointer; color: #475569; font-weight: 500; }
        .btn-run-agent2:hover { background: #eff6ff; border-color: #bfdbfe; color: #2563eb; }
        .grc-status2 { font-size: .75rem; color: #94a3b8; display: flex; align-items: center; gap: 5px; }
        .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #cbd5e1; display: inline-block; }
        .status-dot.running { background: #d97706; animation: pulse2 1s infinite; }
        .status-dot.done { background: #16a34a; }
        .status-dot.error { background: #dc2626; }
        @keyframes pulse2 { 0%,100%{opacity:1} 50%{opacity:.3} }
        .grc-result2 { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px 14px; font-size: .8rem; line-height: 1.7; color: #475569; white-space: pre-wrap; display: none; }
        .grc-result2.visible { display: block; }
        .ctrl-right { display: flex; align-items: center; gap: 8px; margin-left: 8px; }
        .status-sel2 { font-size: .78rem; border: 1px solid #e2e8f0; border-radius: 5px; padding: 5px 26px 5px 10px; color: #475569; background: #f8fafc; cursor: pointer; outline: none; appearance: none; -webkit-appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' fill='%2394a3b8' viewBox='0 0 10 6'%3E%3Cpath d='M0 0l5 6 5-6z'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 8px center; min-width: 120px; }
        .status-sel2.passing { background-color: #f0fdf4; border-color: #bbf7d0; color: #16a34a; }
        .status-sel2.failing { background-color: #fef2f2; border-color: #fecaca; color: #dc2626; }
        .status-sel2.in_progress { background-color: #fffbeb; border-color: #fde68a; color: #d97706; }
        .btn-jira2 { font-size: .78rem; background: #2563eb; color: #fff; border: none; border-radius: 5px; padding: 5px 12px; cursor: pointer; font-weight: 600; white-space: nowrap; }
        .btn-jira2:hover { background: #1d4ed8; }
        .ctrl-expand-btn { font-size: .7rem; color: #94a3b8; background: none; border: none; cursor: pointer; padding: 2px 4px; }
        .ctrl-expand-btn:hover { color: #2563eb; }

        @media (max-width: 900px) {
          .soc-kpis { grid-template-columns: repeat(3,1fr); }
          .soc-sidebar { width: 160px; min-width: 160px; }
        }
      </style>

      <div class="soc-wrap">
        <!-- HEADER -->
        <div class="soc-hdr">
          <div>
            <div class="soc-hdr-title">&#128274; SOC 2 Compliance Dashboard</div>
            <div class="soc-hdr-sub">Track and manage your SOC 2 controls</div>
          </div>
          <button class="btn-sync-jira" onclick="syncJira2()">&#128279; Sync Jira</button>
        </div>

        <!-- KPI CARDS -->
        <div class="soc-kpis">
          <div class="soc-kpi kpi-total"><div class="soc-kpi-num" id="kpi-total2">57</div><div class="soc-kpi-label">TOTAL CONTROLS</div></div>
          <div class="soc-kpi kpi-passing"><div class="soc-kpi-num" id="kpi-passing2">0</div><div class="soc-kpi-label">&#9989; PASSING</div></div>
          <div class="soc-kpi kpi-failing"><div class="soc-kpi-num" id="kpi-failing2">0</div><div class="soc-kpi-label">&#10060; FAILING</div></div>
          <div class="soc-kpi kpi-inprog"><div class="soc-kpi-num" id="kpi-inprog2">0</div><div class="soc-kpi-label">&#9888;&#65039; IN PROGRESS</div></div>
          <div class="soc-kpi kpi-notstarted"><div class="soc-kpi-num" id="kpi-notstarted2">57</div><div class="soc-kpi-label">&#9711; NOT STARTED</div></div>
        </div>

        <!-- PROGRESS BAR -->
        <div class="soc-progress-section">
          <div class="soc-progress-title">Overall Compliance Progress</div>
          <div class="soc-progress-bar">
            <div class="pb-seg pb-passing" id="pb-passing2" style="width:0%"></div>
            <div class="pb-seg pb-inprog" id="pb-inprog2" style="width:0%"></div>
            <div class="pb-seg pb-failing" id="pb-failing2" style="width:0%"></div>
            <div class="pb-seg pb-notstarted" id="pb-notstarted2" style="width:100%"></div>
          </div>
          <div class="soc-progress-legend">
            <span class="prog-legend-item"><span class="prog-dot" style="background:#16a34a"></span>Passing</span>
            <span class="prog-legend-item"><span class="prog-dot" style="background:#d97706"></span>In Progress</span>
            <span class="prog-legend-item"><span class="prog-dot" style="background:#dc2626"></span>Failing</span>
            <span class="prog-legend-item"><span class="prog-dot" style="background:#cbd5e1"></span>Not Started</span>
          </div>
        </div>

        <!-- BODY -->
        <div class="soc-body">
          <!-- SIDEBAR -->
          <div class="soc-sidebar">
            <div class="soc-sb-item active" data-cat="all" onclick="filterGRC2('all',this)">
              <div class="soc-sb-cc">ALL</div>
              <div class="soc-sb-name">All Controls <span class="sb-count"></span></div>
            </div>
            <div class="soc-sb-item" data-cat="cc1" onclick="filterGRC2('cc1',this)">
              <div class="soc-sb-cc">CC1</div><div class="soc-sb-name">Control Environment</div>
            </div>
            <div class="soc-sb-item" data-cat="cc2" onclick="filterGRC2('cc2',this)">
              <div class="soc-sb-cc">CC2</div><div class="soc-sb-name">Communication &amp; Information</div>
            </div>
            <div class="soc-sb-item" data-cat="cc3" onclick="filterGRC2('cc3',this)">
              <div class="soc-sb-cc">CC3</div><div class="soc-sb-name">Risk Assessment</div>
            </div>
            <div class="soc-sb-item" data-cat="cc4" onclick="filterGRC2('cc4',this)">
              <div class="soc-sb-cc">CC4</div><div class="soc-sb-name">Monitoring Activities</div>
            </div>
            <div class="soc-sb-item" data-cat="cc5" onclick="filterGRC2('cc5',this)">
              <div class="soc-sb-cc">CC5</div><div class="soc-sb-name">Control Activities</div>
            </div>
            <div class="soc-sb-item" data-cat="cc6" onclick="filterGRC2('cc6',this)">
              <div class="soc-sb-cc">CC6</div><div class="soc-sb-name">Logical &amp; Physical Access</div>
            </div>
            <div class="soc-sb-item" data-cat="cc7" onclick="filterGRC2('cc7',this)">
              <div class="soc-sb-cc">CC7</div><div class="soc-sb-name">System Operations</div>
            </div>
            <div class="soc-sb-item" data-cat="cc8" onclick="filterGRC2('cc8',this)">
              <div class="soc-sb-cc">CC8</div><div class="soc-sb-name">Change Management</div>
            </div>
            <div class="soc-sb-item" data-cat="cc9" onclick="filterGRC2('cc9',this)">
              <div class="soc-sb-cc">CC9</div><div class="soc-sb-name">Risk Mitigation</div>
            </div>
            <div class="soc-sb-item" data-cat="c1" onclick="filterGRC2('c1',this)">
              <div class="soc-sb-cc">C1</div><div class="soc-sb-name">Confidentiality</div>
            </div>
            <div class="soc-sb-item" data-cat="a1" onclick="filterGRC2('a1',this)">
              <div class="soc-sb-cc">A1</div><div class="soc-sb-name">Availability</div>
            </div>
          </div>

          <!-- PANEL -->
          <div class="soc-panel">
            <div class="soc-search-wrap">
              <input class="soc-search" type="text" placeholder="Search controls..." oninput="searchGRC2(this.value)">
            </div>
            <div class="soc-list-hdr">
              <div class="soc-list-hdr-left">
                <h3 id="cat-title2">All Controls</h3>
                <p id="cat-sub2">All SOC 2 Trust Service Criteria</p>
              </div>
              <span class="soc-list-count" id="cat-count2">57 controls</span>
            </div>
            <div class="soc-list">
''' + rows_html + r'''
            </div>
          </div>
        </div>
      </div>

      <script>
      (function(){
        const SOC2_KEY = 'soc2_statuses';
        const JIRA_KEY = 'jira_base_url';
        function getStatuses(){ return JSON.parse(localStorage.getItem(SOC2_KEY)||'{}'); }

        window.saveStatus2 = function(id, val, selEl) {
          const s = getStatuses(); s[id] = val;
          localStorage.setItem(SOC2_KEY, JSON.stringify(s));
          const sq = document.getElementById('sq-'+id);
          if(sq) sq.className = 'ctrl-status-sq ' + val;
          if(selEl){ selEl.className = 'status-sel2 ' + val; }
          updateGRCKPIs();
        };

        window.updateGRCKPIs = function() {
          const s = getStatuses();
          const cards = document.querySelectorAll('.ctrl-row[data-id]');
          let total=cards.length, passing=0, failing=0, inprog=0, notstarted=0;
          cards.forEach(c => {
            const st = s[c.dataset.id] || 'not_started';
            const sq = document.getElementById('sq-'+c.dataset.id);
            const sel = c.querySelector('.status-sel2');
            if(sq) sq.className='ctrl-status-sq '+st;
            if(sel){ sel.value=st; sel.className='status-sel2 '+st; }
            if(st==='passing') passing++;
            else if(st==='failing') failing++;
            else if(st==='in_progress') inprog++;
            else notstarted++;
          });
          const q = id => document.getElementById(id);
          if(q('kpi-total2')) q('kpi-total2').textContent=total;
          if(q('kpi-passing2')) q('kpi-passing2').textContent=passing;
          if(q('kpi-failing2')) q('kpi-failing2').textContent=failing;
          if(q('kpi-inprog2')) q('kpi-inprog2').textContent=inprog;
          if(q('kpi-notstarted2')) q('kpi-notstarted2').textContent=notstarted;
          const pct = v => total ? (v/total*100).toFixed(1)+'%' : '0%';
          const pb = id => document.getElementById(id);
          if(pb('pb-passing2')) pb('pb-passing2').style.width=pct(passing);
          if(pb('pb-inprog2')) pb('pb-inprog2').style.width=pct(inprog);
          if(pb('pb-failing2')) pb('pb-failing2').style.width=pct(failing);
          if(pb('pb-notstarted2')) pb('pb-notstarted2').style.width=pct(notstarted);
          document.querySelectorAll('.soc-sb-item[data-cat]').forEach(el => {
            const cat = el.dataset.cat;
            const count = cat==='all' ? total : document.querySelectorAll('.ctrl-row[data-cat="'+cat+'"]').length;
            const cb = el.querySelector('.sb-count');
            if(cb) cb.textContent = count;
          });
        };

        window.filterGRC2 = function(cat, el) {
          document.querySelectorAll('.soc-sb-item').forEach(e=>e.classList.remove('active'));
          if(el) el.classList.add('active');
          const rows = document.querySelectorAll('.ctrl-row[data-id]');
          let visible=0;
          rows.forEach(r => {
            const show = cat==='all' || r.dataset.cat===cat;
            r.style.display = show ? '' : 'none';
            if(show) visible++;
          });
          const catNames={all:'All Controls',cc1:'Control Environment',cc2:'Communication & Information',cc3:'Risk Assessment',cc4:'Monitoring Activities',cc5:'Control Activities',cc6:'Logical & Physical Access',cc7:'System Operations',cc8:'Change Management',cc9:'Risk Mitigation',c1:'Confidentiality',a1:'Availability'};
          const catSubs={all:'All SOC 2 Trust Service Criteria',cc1:'COSO Principles 1\u20135',cc2:'COSO Principles 13\u201315',cc3:'COSO Principles 6\u20139',cc4:'COSO Principles 16\u201317',cc5:'COSO Principles 10\u201312',cc6:'Logical and Physical Access Controls',cc7:'System Operations Controls',cc8:'Change Management Controls',cc9:'Risk Mitigation Controls',c1:'Confidentiality Controls',a1:'Availability Controls'};
          const t=document.getElementById('cat-title2'); if(t) t.textContent=catNames[cat]||cat;
          const sub=document.getElementById('cat-sub2'); if(sub) sub.textContent=catSubs[cat]||'';
          const cnt=document.getElementById('cat-count2'); if(cnt) cnt.textContent=visible+' controls';
        };

        window.searchGRC2 = function(val) {
          const q=val.toLowerCase().trim();
          document.querySelectorAll('.ctrl-row[data-id]').forEach(r => {
            r.style.display = !q || r.textContent.toLowerCase().includes(q) ? '' : 'none';
          });
        };

        window.openJira2 = function(title, ccId) {
          let base = localStorage.getItem(JIRA_KEY);
          if(!base){ base=prompt('Enter your Jira base URL (e.g. https://yourcompany.atlassian.net):',''); if(!base) return; localStorage.setItem(JIRA_KEY,base); }
          const origin = base.startsWith('http') ? new URL(base).origin : 'https://'+base;
          window.open(origin+'/secure/CreateIssueDetails!init.jspa?issuetype=10001&summary='+encodeURIComponent('[SOC2 '+ccId+'] '+title),'_blank');
        };

        window.syncJira2 = function() {
          const cur = localStorage.getItem(JIRA_KEY)||'';
          const val = prompt('Jira base URL (e.g. https://yourco.atlassian.net):',cur);
          if(val!==null){ localStorage.setItem(JIRA_KEY,val); }
        };

        window.toggleAgentRow = function(id) {
          const wrap = document.getElementById('agent-wrap-'+id);
          if(wrap) wrap.classList.toggle('expanded');
        };

        document.addEventListener('DOMContentLoaded', updateGRCKPIs);
        setTimeout(updateGRCKPIs, 300);
      })();
      </script>
    </div><!-- /tab-grc -->'''

# ── read original file ────────────────────────────────────────────────────────
filepath = '/Users/zach.ahrak/web-agent/templates/index.html'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# lines are 0-indexed; line 934 = index 933, line 2774 = index 2773
start_idx = 933  # line 934 (0-based)
end_idx   = 2773 # line 2774 (0-based)

new_lines = lines[:start_idx] + [tab_html + '\n'] + lines[end_idx + 1:]

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Done. Replaced lines 934-2774 ({end_idx - start_idx + 1} old lines) with new tab HTML.")
print(f"New file has {len(new_lines)} lines.")
