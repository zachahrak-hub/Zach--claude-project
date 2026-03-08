# Access Control Matrix (CRLG#29, #30, #32, #34, #39)

**Effective Date:** March 8, 2026
**Classification:** Confidential - Internal Use Only

---

## Overview

This document defines role-based access controls (RBAC) across all critical systems and infrastructure.

**Principle:** Least Privilege - All users granted minimum permissions required for role.

---

## 1. OKTA (Authentication & Authorization)

### Roles Defined

| Role | Description | Systems | Permissions |
|------|-------------|---------|-------------|
| **Admin** | Full platform & infrastructure access | All | Full access to all systems |
| **Developer** | Development, staging, and production app access | GitHub, AWS, Database, Okta | Read/write code; deploy to staging; query logs |
| **DevOps Engineer** | Infrastructure management | AWS, GitHub, Database, Okta, Teleport | Manage infrastructure; apply patches; manage backups |
| **Security Engineer** | Security monitoring & incident response | AWS, GitHub, Okta, Incident tools | View all logs; create security incidents; manage MFA |
| **Operations** | Monitor systems & respond to alerts | Monitoring tools, Database | Read-only access; escalate incidents |
| **Contractor** | Limited time-bound access | GitHub, AWS (staging only) | Limited to specific projects; 3-month expiry |

### MFA Requirements

**All users MUST enable MFA:**
- Okta MFA (primary)
- TOTP app (Okta or Google Authenticator) or hardware key
- Backup codes stored securely

**Admin users MUST use hardware keys or high-assurance MFA**

### Access Provisioning

- **New hire:** Within 24 hours of start date
- **Contractor:** Upon contract start with explicit end date
- **Termination:** Within 24 hours of employment end

### Access Review

- **Quarterly:** All access reviewed and updated per role changes
- **Annual:** Complete re-certification by managers
- **On-demand:** Immediate review if role/team changes

---

## 2. AWS (Cloud Infrastructure)

### Permission Structure

**Organization Account Level:**
- Admin: Management of all AWS accounts, IAM, billing
- Security: View-only access to all accounts; create security resources

**Production Account:**
- Admin: Full production access (restricted team)
- Developer: Deploy & query; no delete permissions
- DevOps: Full infrastructure management

**Staging Account:**
- Developer: Full access
- Contractor: Limited access (specific resources only)

### Access Methods

1. **AWS SSO / IAM Identity Center**
   - Primary authentication via Okta
   - Role-based access via permission sets
   - Multi-factor authentication required

2. **Database Access**
   - RDS proxy with IAM authentication
   - No direct password auth allowed
   - Temporary credentials with 1-hour TTL

3. **SSH Access to EC2**
   - Via Teleport (VPN gateway)
   - SSH key pairs managed in AWS Secrets Manager
   - All connections logged

### Privileged Access Management (PAM)

- **Admin access** requires just-in-time (JIT) activation
- Activation logged and requires approval
- Access elevated for maximum 4 hours
- All privileged actions logged to CloudTrail

### Root Account Protection

- Root credentials locked in vault
- Offline backup stored securely
- MFA enabled on root account
- Only accessed for account recovery or root-only operations

---

## 3. GITHUB (Source Code)

### Repository Structure

**Organizations:**
- `coralogix` (main product org)
- `coralogix-internal` (internal tools)
- `coralogix-third-party` (vendor integrations)

### Branch Protection Rules

**Main/Master Branch:**
- Require pull request reviews (minimum 2 approvers)
- Require status checks (tests, security scans)
- Require code approval from designated reviewers
- Dismiss stale pull request approvals
- Restrict force push to admin only

### Roles & Permissions

| Role | Repositories | Permissions | Branch Access |
|------|--------------|-------------|---------------|
| **Admin** | All | Full access; merge; delete; settings | All branches |
| **Owner** | All | Manage team; access control; billing | All branches |
| **Reviewer** | Core product | Read/write; approve PRs | Protected branches only |
| **Developer** | Assigned repos | Create branches; push; open PRs | Feature branches only |
| **Contributor** | Public repos | Fork; create PRs | Via PR only |

### SSH Key Management

- SSH keys required for all repository access
- Keys rotated annually or upon compromise
- Compromised keys revoked immediately
- Key usage logged and audited

### Secrets Protection

- No secrets committed to code (pre-commit hooks scan)
- Secrets stored in GitHub Secrets or AWS Secrets Manager
- Automatic revocation if secrets detected
- Code scanning enabled on all repositories

---

## 4. DATABASE ACCESS (CRLG#33)

### Users & Roles

**Production Database (mysql-prod):**
- `app_user` → Application queries (read/write limited tables)
- `read_only` → Analytics and monitoring (select only)
- `dba_user` → Database administration (restricted to DevOps)

**Staging Database:**
- `dev_user` → Full read/write for development
- `test_user` → Limited access for testing

### Access Control

- **Network:** Only from application tier (security group restrictions)
- **Authentication:** IAM authentication (temporary credentials)
- **Encryption:** All connections use TLS 1.2+
- **Logging:** All connections logged to CloudWatch

### Privilege Separation

- Application user: read/write limited to business logic tables
- DBA user: requires approval and 4-hour time window
- No user has delete permissions on production data

---

## 5. REMOTE ACCESS / VPN (CRLG#39)

### Teleport (VPN Gateway)

**Access Method:**
1. Employee authenticates via Okta + MFA
2. Teleport verifies Okta identity
3. Issues temporary SSH certificate (valid 1 hour)
4. SSH session logged to CloudTrail + local logs

**Configuration:**
- Only authenticated via Okta
- Restricted to approved IP ranges (optional)
- Automatic session recording
- Activity available for audit

**Access Levels:**
- **Developers:** Access to staging/dev systems
- **DevOps:** Access to all systems with logging
- **Security:** Full access with heightened monitoring

---

## 6. OKTA ADMIN ACCESS (CRLG#32)

### Okta Super Admin

**Current Super Admins:** (List maintained separately in Okta)

**Restrictions:**
- No more than 2 active super admins
- MFA required on all logins
- Admin activities logged automatically
- Quarterly review of admin actions

### Okta Admin Roles

| Role | Permissions | Count |
|------|-------------|-------|
| Super Admin | Full Okta access | ≤2 |
| Org Admin | Organization management | ≤5 |
| Security Admin | SSO/MFA/security policies | ≤3 |
| Group Admin | Group membership management | Variable |
| User Admin | User provisioning/deprovisioning | ≤2 |

---

## 7. SEGREGATION OF DUTIES

**Principle:** No single person should have full control over critical functions.

### Enforced Separations

| Function | Admin Role | Reviewer Role |
|----------|-----------|---------------|
| **Code Deployment** | DevOps Engineer | Release Manager approval required |
| **Infrastructure Changes** | DevOps Engineer | Security review before production |
| **Access Approval** | Not accessible to requester | Manager approval required |
| **Incident Response** | Incident Commander | Security Officer oversight |
| **Financial/Billing** | Finance Admin | CFO approval for large changes |

---

## 8. CONTRACTOR & TEMPORARY ACCESS

**Default Duration:** 3 months with renewal option

**Onboarding:**
1. Contract reviewed and signed
2. NDA/Confidentiality agreement signed
3. Access provisioned for specific systems only
4. Okta group added with expiration date
5. Manager notified of access level

**Restrictions:**
- Access limited to non-sensitive systems
- No production admin access
- No access to customer data
- No VPN access unless approved
- All access logged for audit

**Offboarding:**
1. Manager notifies IT of termination
2. Access revoked within 24 hours
3. Keys/tokens deactivated
4. Exit interview conducted
5. Equipment returned

---

## 9. QUARTERLY ACCESS REVIEW (CRLG#35)

**Q1 Review (Jan-Mar):**
- All user access across all systems audited
- Orphaned accounts identified and removed
- Role changes verified
- Manager sign-off required

**Process:**
1. IT generates access report for each manager
2. Manager reviews and certifies accuracy
3. IT remediates any exceptions
4. Security team spot-checks compliance
5. Results documented and archived

---

## 10. INCIDENT RESPONSE ACCESS

**During Security Incident:**

- **Incident Commander:** Full access to all systems (time-limited)
- **Forensics Team:** Read-only access to logs and data
- **Communications Lead:** Access to notification systems
- **All access logged and reviewed post-incident**

---

## Compliance & Monitoring

**Automated Checks:**
- Weekly: Missing MFA enforcement
- Weekly: Inactive account identification
- Monthly: Permission consistency audits
- Quarterly: Access certification reviews

**Tools:**
- Okta: User provisioning & authentication logs
- AWS: IAM Access Analyzer & CloudTrail
- GitHub: Audit logs & branch protection verification

---

## Approval & Accountability

**Policy Owner:** Chief Information Security Officer
**Last Updated:** March 8, 2026
**Next Review:** June 8, 2026

All managers must acknowledge quarterly access reviews.
All users must acknowledge receipt of this policy annually.

---

*Document Classification: Confidential*
