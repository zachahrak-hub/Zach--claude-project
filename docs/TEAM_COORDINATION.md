# SOC 2 Team Coordination Guide

**Document:** Team coordination checklist for YELLOW controls requiring special handling
**Effective Date:** March 8, 2026

---

## Overview

Three controls require special team coordination before evidence collection can be automated:

1. **CRLG#19** — Customer Support Tickets (needs CS team)
2. **CRLG#26** — System Monitoring & Alerts (needs DevOps/Platform team)
3. **CRLG#36** — New Employee Access (needs HR team)

This document outlines the coordination needed for each.

---

## CRLG#19: Customer Support Service Level Tracking

### Requirement (from SOC 2)
> Customer Support team personnel receive telephone and email for customer support. Support team follows defined protocols for recording, resolving, and escalating received requests. Calls are documented in Help desk system. Support team meets SLA commitment.

### Evidence Needed
1. Log tracker report from support system (FAST/Intercom) for audit period
2. Sample of support tickets showing: ticket ID, creation time, first response time, resolution time
3. Verification that response/resolution times meet SLA commitments

### Coordination with CS Team

**Kickoff Meeting:**
- [ ] Schedule with: Customer Success Lead
- [ ] Discuss: Current support ticket tracking system (Intercom, Zendesk, etc.)
- [ ] Confirm: SLA targets (response time, resolution time)
- [ ] Define: How to query support tickets for audit period

**Technical Details to Clarify:**
1. **Support System:** Which tool tracks support tickets?
   - Intercom / Zendesk / Custom system?
   - Data export format available?
   - API access available?

2. **SLA Definition:**
   - First response time target (hours)?
   - Resolution time target (days)?
   - Priority levels affecting SLA?

3. **Sample Selection:**
   - How many tickets to sample? (20-50 recommended)
   - Random selection or specific criteria?
   - Include all priority levels?

4. **Query Methodology:**
   - Filter by date range (audit period)
   - Filter by status: "Closed" / "Resolved"
   - Include: ticket ID, created date, first response date, resolved date, customer

5. **Report Generation:**
   - Can CS team provide automated report?
   - Manual export from support system?
   - API query for extraction?

### Action Items

**CS Team Responsibilities:**
- [ ] Confirm SLA targets documented
- [ ] Provide access to support system for query
- [ ] Export sample of closed tickets for audit period
- [ ] Verify ticket data includes required fields (timestamps, resolution status)

**Audit Team Responsibilities:**
- [ ] Extract sample tickets
- [ ] Verify response times vs SLA
- [ ] Document any SLA violations
- [ ] Prepare evidence for auditor

### Timeline
- **Coordination:** Before audit evidence collection
- **Sample extraction:** 1-2 weeks before audit deadline
- **Verification:** 1 week before audit deadline

---

## CRLG#26: System Monitoring & Alerts Configuration

### Requirement (from SOC 2)
> DevOps operations monitor system performance, security threats, changing resource utilization needs, and unusual system activity. Alerts are generated and sent to DevOps staff. Alert tracking documented until resolved.

### Evidence Needed
1. Monitoring dashboard showing all systems are monitored
2. Alert configuration (what triggers alerts, where alerts sent)
3. Proof that alerts are received by DevOps team
4. Sample of resolved alerts showing issue was addressed

### Coordination with DevOps/Platform Team

**Kickoff Meeting:**
- [ ] Schedule with: DevOps Lead / Platform Engineer
- [ ] Discuss: Current monitoring stack (Prometheus, DataDog, CloudWatch?)
- [ ] Confirm: What systems are monitored
- [ ] Define: Alert thresholds and recipients

**Technical Details to Clarify:**
1. **Monitoring Stack:**
   - Primary monitoring tool? (CloudWatch, Prometheus, DataDog, New Relic?)
   - Secondary tools? (Grafana, custom dashboards?)
   - All critical systems monitored?

2. **Alert Configuration:**
   - CPU/Memory alerts?
   - Disk space alerts?
   - Network latency alerts?
   - Security/attack alerts?
   - Application error rate alerts?
   - Database performance alerts?

3. **Alert Routing:**
   - Where are alerts sent? (PagerDuty, Slack #alerts, email, SMS?)
   - Who receives which alerts?
   - Escalation procedures?

4. **Alert Response:**
   - How are alerts tracked (ticketed)?
   - Response time targets?
   - Root cause analysis process?
   - How is resolution documented?

5. **Evidence Collection:**
   - Screenshot of monitoring dashboard
   - Alert rule export/configuration
   - Sample of recent alerts with resolution
   - Alert metrics (% resolved within SLA)

### Action Items

**DevOps Team Responsibilities:**
- [ ] Provide screenshot of monitoring dashboard
- [ ] Export alert configuration as YAML/JSON
- [ ] Confirm all critical systems have alerts
- [ ] Provide sample of recent alerts with resolutions
- [ ] Document alert SLA (response time)

**Audit Team Responsibilities:**
- [ ] Review monitoring coverage
- [ ] Verify alert configuration is complete
- [ ] Spot-check alert responses vs SLA
- [ ] Prepare evidence for auditor

### Timeline
- **Coordination:** ASAP - ongoing process
- **Dashboard screenshot:** 1 week before audit
- **Alert sample:** 1 week before audit
- **Verification:** Final week before audit

### Current Status
- ✅ Partial implementation in place (Audit logs active)
- ⏳ Needs: Full monitoring stack documentation
- ⏳ Needs: Alert routing configuration

---

## CRLG#36: New Employee Access Verification

### Requirement (from SOC 2)
> New access to network, systems and software is authorized and granted by appropriate individuals. For selected sample of new employees, provide evidence showing they received access to required systems.

### Evidence Needed
1. List of new employees hired during audit period
2. Sample selected by auditor (typically 10-20% of new hires)
3. For each sampled employee:
   - Proof they received onboarding
   - Evidence they received system access (username provisioned)
   - Verification of which systems they access

### Coordination with HR Team

**Kickoff Meeting:**
- [ ] Schedule with: HR Manager / People Operations
- [ ] Discuss: New hire onboarding process
- [ ] Confirm: Systems new hires need access to
- [ ] Define: How to verify access was granted

**Technical Details to Clarify:**
1. **New Hire Tracking:**
   - Where is new hire data stored? (Bamboo HR, Workday, ADP?)
   - Can query for new hires by date range?
   - Data available: name, start date, role, manager?

2. **Onboarding Process:**
   - Formal onboarding checklist used?
   - Where documented? (Jira, Asana, Google Docs?)
   - Who verifies completion?
   - Training tracking system? (Litmus, custom?)

3. **System Access Provisioning:**
   - Okta provisioning workflow?
   - Manual provisioning by IT?
   - Access request approval system?
   - How is provisioning verified?

4. **System Access Verification:**
   - Which systems must new hires have? (GitHub, AWS, Database, Email, etc.)
   - How to verify access was granted?
   - Which system has authoritative access list?
   - Onboarding completion tracking?

5. **Sample Documentation:**
   - Onboarding checklist signed/completed
   - Access request tickets
   - Proof of account creation (screenshot of directory listing)
   - Training completion evidence

### Action Items

**HR Team Responsibilities:**
- [ ] Provide list of new hires for audit period
- [ ] When auditor selects sample, provide:
  - [ ] Onboarding checklists (signed)
  - [ ] Training completion documentation
  - [ ] Start date verification
  - [ ] Manager name & approval

**IT Team Responsibilities:**
- [ ] When sample names provided:
  - [ ] Verify Okta accounts exist for each
  - [ ] Verify GitHub access provisioned
  - [ ] Verify AWS/database access provisioned
  - [ ] Export access proof (screenshots of user lists)

**Audit Team Responsibilities:**
- [ ] Extract new hire list from HR
- [ ] Provide sample to auditor for selection
- [ ] Collect evidence from IT & HR
- [ ] Cross-reference onboarding & access provisioning
- [ ] Prepare evidence for auditor

### Timeline
- **Coordination:** Before audit begins
- **New hire list extraction:** 2 weeks before audit
- **Auditor sample selection:** 1 week before audit deadline
- **Access verification:** 1 week before audit deadline
- **Evidence collection:** Final days before audit

### Current Status
- ⏳ Not yet coordinated
- ⏳ Needs: HR meeting to discuss onboarding tracking
- ⏳ Needs: IT process for verifying access provisioning
- ⏳ Needs: Automated export of new hire + access data

---

## COORDINATION TIMELINE

### Week 1 (This Week)
- [ ] Schedule kickoff calls with all three teams
- [ ] Share this coordination document
- [ ] Discuss timeline and responsibilities

### Week 2-3
- [ ] CS Team: Export sample support tickets
- [ ] DevOps Team: Provide monitoring dashboard & alert config
- [ ] HR Team: Provide new hire tracking process

### Week 4
- [ ] Verify sample data is complete
- [ ] Identify any gaps or issues
- [ ] Escalate blockers

### Week 5
- [ ] Final evidence collection
- [ ] Auditor sample selection (HR)
- [ ] Prepare evidence package

---

## Contact Information

### Team Leads
- **CS Team Lead:** [Name, Email, Phone]
- **DevOps Lead:** [Name, Email, Phone]
- **HR Manager:** [Name, Email, Phone]
- **CISO:** [Name, Email, Phone]

### Coordination Owners
- **CRLG#19 (CS):** [Owner Name]
- **CRLG#26 (DevOps):** [Owner Name]
- **CRLG#36 (HR):** [Owner Name]
- **Overall Audit Coordinator:** [CISO or GRC role]

---

## Next Steps

1. **Share this document** with all team leads
2. **Schedule coordination calls** with each team
3. **Assign owners** for each control
4. **Set up tracking** in Jira for all coordination tasks
5. **Monthly check-ins** to ensure progress

---

*Document Classification: Confidential*
