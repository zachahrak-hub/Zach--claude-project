# Change Management & Code Review Process (CRLG#52, #53, #54)

**Effective Date:** March 8, 2026
**Classification:** Internal

---

## Overview

This document defines the controlled process for all changes to systems, applications, and infrastructure. All changes require code review, testing, and approval before deployment to production.

---

## 1. CHANGE REQUEST WORKFLOW

### Step 1: Change Proposal & Design

**Who:** Developer or DevOps Engineer
**Tools:** Jira + GitHub

**Requirements:**
- Create a Jira ticket describing the change
- Include: problem statement, proposed solution, risk assessment
- Link to any related tickets or documentation
- Assign severity: Patch, Minor, Major, Critical

**Change Categories:**
- **Security Patch:** CVE fix, vulnerability remediation
- **Bug Fix:** Code correction, performance improvement
- **Feature:** New functionality or enhancement
- **Infrastructure:** Environment changes, config updates
- **Dependency:** Library or framework updates

### Step 2: Code Implementation & Version Control

**Branch Strategy:**

```
main (production-ready)
  ├── feature/TICKET-123-description  (feature branch)
  ├── bugfix/TICKET-456-description   (bugfix branch)
  ├── hotfix/TICKET-789-description   (hotfix branch - critical)
  └── refactor/TICKET-321-description (refactor branch)
```

**Commit Requirements:**
- Meaningful commit messages with ticket reference: `TICKET-123: Description`
- One feature per commit (atomic commits)
- No hardcoded secrets, credentials, or sensitive data
- Pass pre-commit hooks (linting, secret scanning)

### Step 3: Automated Testing

**Required Checks Before PR Merge:**

1. **Unit Tests**
   - Minimum 70% code coverage
   - All new code paths covered
   - Tests pass in CI pipeline

2. **Integration Tests**
   - Critical user workflows tested
   - Database schema changes verified
   - API contracts validated

3. **Security Scanning**
   - SAST (Static Analysis Security Testing)
   - Dependency vulnerability scanning
   - Secret detection
   - No blocking issues

4. **Code Quality**
   - Lint checks pass
   - Code style consistent with standards
   - No major code smells

**Failure Policy:** Changes MUST NOT merge if any check fails. Manual override only with security approval.

### Step 4: Code Review & Approval

**Pull Request Requirements:**

- **Reviewers:** Minimum 2 approvals required
- **Reviewer Rules:**
  - Code authors cannot approve their own code
  - At least 1 reviewer must be senior engineer
  - For security changes: Security team review required
  - For infra changes: DevOps/SRE review required

**Code Review Checklist:**

```
□ Code solves the stated problem
□ Design is appropriate and maintainable
□ No security vulnerabilities introduced
□ Performance implications considered
□ Edge cases handled properly
□ Tests are adequate and meaningful
□ Documentation updated if needed
□ Breaking changes clearly communicated
□ No hardcoded secrets or sensitive data
□ Follows project coding standards
```

**Review Comments:**
- Reviewer provides feedback
- Author addresses all comments
- Conversations resolved before approval
- Stale approvals dismissed if new changes made

### Step 5: Approval & Authorization

**Approval Required From:**
- 2 code reviewers (peer review)
- Release Manager (deployment authorization)

**Escalation:** If reviewers cannot reach consensus:
- Senior engineer arbitrates
- CISO involved if security-related

**Approval Gate:** PR cannot merge until all approvals obtained.

### Step 6: Deployment Planning

**Staging Deployment First:**
1. Merge PR to `develop` or `staging` branch
2. Deploy to staging environment
3. Manual testing performed
4. Stakeholders notified

**Deployment Planning:**
- Schedule maintenance window if needed
- Notify users of upcoming changes
- Prepare rollback plan
- Test rollback procedure

---

## 2. DEPLOYMENT PROCESS

### Pre-Deployment Checklist

- [ ] All tests passing in CI/CD
- [ ] Code reviewed and approved
- [ ] Staging tested and verified
- [ ] Rollback plan documented and tested
- [ ] Change advisory board (CAB) approval obtained
- [ ] Stakeholder notifications prepared
- [ ] Monitoring alerts configured
- [ ] On-call engineer available

### Deployment Steps

1. **Pre-deployment Notification** (30 minutes before)
   - Slack: `#deployment` channel
   - Email: stakeholders
   - Status page: maintenance scheduled

2. **Deployment Execution**
   - DevOps engineer initiates deployment
   - CI/CD pipeline handles automation
   - Deployment logged in Jira & deployments channel
   - Real-time monitoring during deployment

3. **Verification**
   - Health checks pass
   - Key metrics normal
   - Error rates acceptable
   - Customer reports monitored

4. **Post-Deployment Notification**
   - Slack: deployment completed
   - Status page: updated
   - Stakeholders notified
   - Metrics shared

### Rollback Procedure

**Trigger Conditions:**
- Error rate > 5%
- Response time > 2x baseline
- Core functionality broken
- Data integrity issues detected
- Security incident discovered

**Rollback Steps:**
1. Page on-call engineer
2. Incident created (Pagerduty + Slack)
3. Deployment reversed to previous version
4. Verification of rollback success
5. Root cause analysis initiated
6. Post-mortem scheduled

---

## 3. DEPLOYMENT FREQUENCY & WINDOWS

### Production Deployments

**Frequency:** 2-3 deployments per week typical

**Approved Deployment Windows:**

| Window | Days | Hours | Use Case |
|--------|------|-------|----------|
| **Standard** | Mon-Thu | 9 AM - 5 PM UTC | Regular changes |
| **Extended** | Mon-Thu | 9 AM - 8 PM UTC | Larger changes |
| **Hotfix** | Any | Any | Critical security/production issues |
| **Weekend/Holiday** | No | No | Only critical production issues (approval required) |

### Emergency/Hotfix Deployments

**Criteria:** Production outage, security breach, data loss

**Fast-Track Approval:**
1. CISO or VP Eng approval required
2. Still requires code review (parallel if needed)
3. Immediate deployment authorized
4. Post-deployment review scheduled

---

## 4. CHANGE DOCUMENTATION

### Jira Ticket Requirements

Every change tracked in Jira with:
- **Description:** What is changing and why
- **Risk Assessment:** Potential impact areas
- **Testing Plan:** How change is tested
- **Rollback Plan:** How to reverse if needed
- **Performance Impact:** Any performance considerations
- **Dependencies:** Linked tickets/systems affected
- **Breaking Changes:** Any API or contract changes

### Pull Request Details

- **Description:** Link to Jira, clear explanation
- **Design:** Architecture/design decisions
- **Testing:** Summary of test coverage
- **Verification:** Steps to verify change
- **Breaking Changes:** Clear notice if applicable

### Deployment Notes

- **When:** Date, time, timezone
- **Who:** Engineer name
- **What:** Version/commit deployed
- **Result:** Success/rollback with reason
- **Duration:** How long deployment took
- **Monitoring:** Key metrics post-deployment

---

## 5. CHANGE TRACKING & AUDIT

### Deployment Log

All deployments logged in:
1. **Slack:** `#deployments` channel (automatic)
2. **Jira:** Deployment field updated
3. **CloudTrail:** AWS infrastructure changes
4. **GitHub:** Commit history with tags
5. **Monitoring:** Deployment markers in dashboards

### Audit Trail

**Access to change history:**
- Jira: View all changes and approvals
- GitHub: Complete commit history
- Slack: Deployment notifications (2-year retention)
- CloudTrail: Infrastructure changes (90 days retention)

**Compliance Queries:**
- Who deployed what and when
- What changed and why
- Who approved the change
- Test results before deployment
- Any issues or rollbacks

---

## 6. APPROVAL AUTHORITIES

### Role Definitions

| Role | Authority | For |
|------|-----------|-----|
| **Release Manager** | Approve deployments to prod | All changes |
| **Security CISO** | Approve security changes | Security/auth/encryption changes |
| **VP Engineering** | Escalate approval decisions | Policy changes, architecture decisions |
| **DevOps Lead** | Infrastructure changes | AWS, database, infrastructure |
| **Senior Engineer** | Code review arbitration | Disputed reviews, complex changes |

### Approval Matrix

| Change Type | Code Review | Security | DevOps | RM | VP |
|------------|------------|----------|--------|----|----|
| Bug fix | 2 | - | - | ✓ | - |
| Feature | 2 | - | - | ✓ | - |
| Security patch | 2 | ✓ | - | ✓ | - |
| Infra change | 1 | ✓ | ✓ | ✓ | ✓ |
| Hotfix | 1 | ✓ | ✓ | ✓ | - |
| Breaking change | 2 | - | - | ✓ | ✓ |

---

## 7. METRICS & MONITORING

**Change Metrics Tracked:**
- Deployment frequency (deployments/week)
- Change lead time (days from request to production)
- Change failure rate (% of deployments requiring rollback)
- Mean time to recovery (MTTR) from failures
- Code review time (hours from PR to approval)

**Monthly Review:**
- All metrics reported to leadership
- Trends analyzed for improvement
- Bottlenecks identified and addressed

---

## 8. TRAINING & ONBOARDING

**New Developers:**
1. Complete Git/GitHub training
2. Learn code review standards
3. Observe 2-3 peer code reviews
4. Review code under supervision
5. Perform first change under guidance

**Required Reading:**
- This Change Management document
- Code Review Best Practices
- Security Development Guide
- Incident Response Procedures

---

## 9. EXCEPTIONS & OVERRIDE PROCEDURES

### Documented Exceptions

**Non-Production Environments:**
- Staging: Relaxed review requirements (1 reviewer)
- Development: Minimal requirements
- Personal Dev: No restrictions

**Emergency Changes:**
- Critical production outage: Hotfix track (requires post-review)
- Security breach: CISO override (post-deployment review)
- Data loss: Rollback only (no review needed)

**Exception Log:**
- All exceptions documented in Jira
- Reason recorded
- Post-mortem conducted
- Process improvements noted

---

## 10. POLICY COMPLIANCE & VIOLATIONS

### Monitoring

**Automated Checks:**
- GitHub branch protection enforces review requirement
- CI/CD blocks deploy if tests fail
- Slack alerts on policy violations
- Weekly compliance report generated

### Violations

**Unauthorized Deployments:**
- Immediate rollback
- Incident investigation
- Policy review with team
- Potential disciplinary action

**Insufficient Testing:**
- Change flagged in review
- Code review returns for revisions
- Cannot merge until tests sufficient

---

## Approval & Effective Date

**Policy Owner:** VP Engineering
**Last Updated:** March 8, 2026
**Next Review:** June 8, 2026

**Acknowledgment Required:** All engineers must acknowledge annually

---

*Document Classification: Internal*
