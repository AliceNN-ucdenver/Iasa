# AI-Assisted Software Development Workshop: Complete Implementation Guide

## Executive Summary

This comprehensive workshop guide covers the transition from traditional software development to AI-assisted engineering using Claude Code, GitHub Copilot, and security-first practices. The workshop emphasizes the spectrum from "vibe coding" to production-grade agentic development, with extensive focus on security through CodeQL and Snyk integration. All examples include specific prompts, security considerations, and OWASP compliance.

---

## Part 1: The AI Development Spectrum

### Understanding the Three Paradigms

| Paradigm | Primary Goal | Verification Model | Security Posture | Best For |
|----------|--------------|-------------------|------------------|----------|
| **Vibe Coding** | Speed & Exploration | Verify large chunks post-generation | Minimal, reactive | Prototypes, POCs |
| **AI-Assisted Engineering** | Productivity with Quality | Continuous small verifications | Proactive, integrated | Production features |
| **Agentic Coding** | Autonomous Task Completion | Plan → Execute → Verify → Report | Policy-driven, automated | Well-defined tasks, refactoring |

### The 70% Problem and Human 30%

AI excels at the "accidental complexity" (boilerplate, patterns, common implementations) but struggles with "essential complexity" (domain logic, security decisions, architectural choices). The human 30% includes:

- **Security Architecture**: Threat modeling, authentication flows, data classification
- **Performance Optimization**: Algorithm selection, caching strategies, database indexing
- **Error Handling**: Business-specific error taxonomies, graceful degradation
- **Compliance**: GDPR, HIPAA, PCI-DSS specific requirements
- **User Experience**: Edge cases, accessibility, internationalization

---

## Part 2: Security-First Prompt Engineering

### Core Security Prompt Pattern

```
CONTEXT + SECURITY_REQUIREMENTS + CONSTRAINTS + VALIDATION + TESTS
```

### OWASP-Focused Prompt Templates

#### Template 1: Injection Prevention (SQL/NoSQL/Command)

**Claude Code:**
```markdown
Role: You are a security engineer implementing OWASP injection prevention.

Context: PostgreSQL 14, Node.js 18, TypeScript, Express 4.x, using node-postgres.

Task: Implement a user search function that prevents SQL injection.

Security Requirements:
- Use parameterized queries exclusively
- Input validation with allowlisting (alphanumeric + specific chars)
- Length limits (max 255 chars)
- Escape special characters if displayed
- Rate limiting (10 requests/minute)
- Audit logging for suspicious patterns

Implementation:
```typescript
interface UserSearchParams {
  query: string;
  limit?: number;
  offset?: number;
}
```

Constraints:
- No string concatenation for SQL
- Use pg.query with $1, $2 placeholders
- Validate all inputs with Joi or Zod
- Return typed results
- Log queries with sanitized params only

Tests required:
1. Valid search returns results
2. SQL injection attempts (' OR '1'='1) are sanitized
3. NoSQL injection attempts ({$gt: ""}) fail validation
4. Command injection (;rm -rf /) is blocked
5. Unicode/encoding attacks handled
6. Null byte injection prevented

Generate implementation with full test suite.
```

**GitHub Copilot:**
```markdown
@workspace #codebase

Create a secure user search endpoint following OWASP A03:2021 (Injection).

Requirements:
- Parameterized queries only
- Input validation: ^[a-zA-Z0-9\s\-\.]+$
- Max length: 100 chars
- Use our db.query wrapper (see src/db/client.ts)
- Add rate limiting middleware
- Include tests for:
  - Normal searches
  - SQL injection: ' OR '1'='1
  - NoSQL payloads: {"$gt": ""}
  - Command injection: $(whoami)
  - Null bytes: search%00.php
  - Unicode: search\u0027

Follow our patterns in src/api/users/*.ts
```

**ChatGPT (VS Code):**
```markdown
You're a security architect. Implement a search function resistant to injection attacks.

Given our stack:
- Express + TypeScript
- PostgreSQL with node-postgres
- Existing validation middleware in src/middleware/validate.ts

Create:
1. Zod schema with strict validation
2. Repository method using parameterized queries
3. Controller with rate limiting
4. Comprehensive tests including OWASP injection vectors

Security checklist:
□ No string concatenation in queries
□ Prepared statements only
□ Input allowlisting
□ Output encoding
□ Error messages don't leak schema
□ Audit logging for suspicious inputs

Show implementation in this order: Schema → Repository → Controller → Tests
```

#### Template 2: Authentication & Session Management

**Claude Code:**
```markdown
Role: Security engineer implementing OWASP A07:2021 (Identification and Authentication Failures)

Task: Create secure authentication with these protections:

Security Requirements:
1. Password Requirements:
   - Minimum 12 characters
   - Check against common passwords list
   - bcrypt with cost factor 12
   - No password in response/logs

2. Session Security:
   - Secure, httpOnly, sameSite cookies
   - CSRF tokens
   - Session rotation on privilege change
   - Absolute timeout (8 hours)
   - Idle timeout (30 minutes)

3. Brute Force Protection:
   - Account lockout after 5 failures
   - Progressive delays (2^n seconds)
   - CAPTCHA after 3 attempts
   - IP-based rate limiting

4. Multi-Factor:
   - TOTP support
   - Backup codes
   - Remember device option (30 days)

Implementation plan:
1. Create user model with secure password storage
2. Implement login with all checks
3. Session management with Redis
4. MFA enrollment/verification
5. Comprehensive test suite

Start with the plan, then implement each component.
```

#### Template 3: Access Control & Authorization

**GitHub Copilot Agent Mode:**
```markdown
#codebase Implement OWASP A01:2021 (Broken Access Control) compliant authorization

Create a role-based access control system with:

Security Requirements:
- Principle of least privilege
- Deny by default
- Central authorization checks
- No client-side access decisions
- Audit all access attempts

Implementation:
1. Roles: admin, editor, viewer
2. Resources: documents, users, settings
3. Permissions matrix:
   - admin: all operations
   - editor: create, read, update documents
   - viewer: read documents only

Protections needed:
- Vertical privilege escalation prevention
- Horizontal access control (users can't access other users' resources)
- Indirect object reference checks
- Anti-CSRF tokens
- Rate limiting per role

Generate:
- Middleware for authorization
- Decorators/annotations for routes
- Tests for each attack vector:
  - Accessing admin endpoints as user
  - Modifying other users' data
  - IDOR attacks
  - Missing authorization checks
  - JWT manipulation

Use our existing JWT middleware in src/middleware/auth.ts
```

### Advanced Prompt Patterns for Security

#### Chain-of-Thought Security Analysis

```markdown
Before implementing, think step-by-step about security:

1. What are the input vectors?
2. What validation is needed for each input?
3. What are potential attack scenarios?
4. How would an attacker try to bypass each control?
5. What logging/monitoring is needed?
6. What would indicate an attack in progress?

Now implement with these considerations addressed.
```

#### Few-Shot Security Examples

```markdown
Here are examples of secure vs insecure patterns:

INSECURE:
```javascript
const query = `SELECT * FROM users WHERE id = ${userId}`;
const user = await db.query(query);
```

SECURE:
```javascript
const query = 'SELECT * FROM users WHERE id = $1';
const values = [userId];
const user = await db.query(query, values);
```

INSECURE:
```javascript
const html = `<div>${userInput}</div>`;
```

SECURE:
```javascript
import DOMPurify from 'isomorphic-dompurify';
const html = `<div>${DOMPurify.sanitize(userInput)}</div>`;
```

Now apply these patterns to implement [specific feature].
```

---

## Part 3: CodeQL Integration and Security Scanning

### Setting Up CodeQL with AI-Generated Code

#### CodeQL Configuration for AI Code Review

**.github/workflows/codeql-analysis.yml:**
```yaml
name: "CodeQL Security Analysis"

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '30 1 * * 0'

jobs:
  analyze:
    name: Analyze
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write

    strategy:
      fail-fast: false
      matrix:
        language: [ 'javascript', 'typescript', 'python' ]

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: ${{ matrix.language }}
        # Use security-extended for more comprehensive analysis
        queries: security-extended
        # Custom queries for AI-specific patterns
        config-file: ./.github/codeql/codeql-config.yml

    - name: Autobuild
      uses: github/codeql-action/autobuild@v3

    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3
      with:
        category: "/language:${{matrix.language}}"
        # Enable AI-powered autofix suggestions
        add-autofix-comment: true
```

#### Custom CodeQL Queries for AI-Generated Patterns

**.github/codeql/ai-patterns.ql:**
```ql
/**
 * @name AI-Generated SQL Injection Risk
 * @description Detects patterns common in AI-generated SQL code
 * @kind problem
 * @problem.severity error
 * @security-severity 9.0
 * @tags security
 *       external/cwe/cwe-089
 *       ai-generated
 */

import javascript
import semmle.javascript.security.dataflow.SqlInjection

class AIGeneratedSQLPattern extends DataFlow::Configuration {
  AIGeneratedSQLPattern() { 
    this = "AIGeneratedSQLPattern" 
  }

  override predicate isSource(DataFlow::Node source) {
    exists(Comment c |
      c.getText().matches("%AI-generated%") or
      c.getText().matches("%Generated by%") or
      c.getText().matches("%Copilot%")
    ) and
    source.asExpr() instanceof StringLiteral
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(MethodCallExpr mce |
      mce.getMethodName() = ["query", "execute", "run"] and
      sink.asExpr() = mce.getArgument(0)
    )
  }
}

from AIGeneratedSQLPattern cfg, DataFlow::PathNode source, DataFlow::PathNode sink
where cfg.hasFlowPath(source, sink)
select sink.getNode(), source, sink, 
  "Potential SQL injection in AI-generated code from $@.", 
  source.getNode(), "user input"
```

### Prompts for CodeQL-Compliant Code

**Claude Code:**
```markdown
Generate a data access layer that will pass CodeQL analysis:

Requirements:
1. No CWE-89 (SQL Injection) vulnerabilities
2. No CWE-79 (XSS) vulnerabilities  
3. No CWE-502 (Deserialization) issues
4. Pass security-extended query suite

Implement:
- User CRUD operations
- Search with filters
- Bulk operations
- Transaction support

Include:
- Parameterized queries only
- Input validation layer
- Output encoding
- Error handling without info leakage
- Audit logging

Add comments: // CodeQL: Safe - [reason] for security decisions
```

---

## Part 4: Snyk Integration for Continuous Security

### Snyk Configuration for AI Development

#### .snyk Configuration File

```yaml
# Snyk configuration for AI-generated code
version: v1.0.0
language-settings:
  javascript:
    enableLinters: true
  python:
    enableLinters: true
    
# Custom rules for AI-generated code patterns
rules:
  - id: ai-hardcoded-secret
    description: Detect potential hardcoded secrets in AI-generated code
    severity: high
    pattern: |
      (?i)(api[_-]?key|secret|token|password|pwd|auth|credential)[\s]*[:=][\s]*["'][^"']{8,}["']
      
  - id: ai-unsafe-deserialization
    description: Prevent unsafe deserialization patterns
    severity: critical
    files:
      - "**/*.js"
      - "**/*.ts"
    pattern: |
      (JSON\.parse|eval|Function|deserialize)\s*\([^)]*user|req|input|data[^)]*\)

# Ignore false positives in test files
exclude:
  - "**/*test*"
  - "**/mock*"
  - "**/.github/**"

# Monitor dependencies for AI code generation libraries
monitor:
  - openai
  - anthropic
  - "@anthropic/claude-sdk"
  - "github-copilot"
  
# Fail builds on high severity issues
failOn: high
```

#### Snyk CLI Integration Prompts

**Claude Code with Snyk:**
```markdown
Implement a file upload handler that passes Snyk security checks:

Pre-implementation scan:
```bash
snyk test --severity-threshold=high
snyk code test
```

Requirements:
- File type validation (images only: jpg, png, webp)
- Size limits (5MB max)
- Virus scanning integration point
- Secure storage (outside webroot)
- No path traversal vulnerabilities
- Generate unique filenames
- Validate magic bytes, not just extension

Post-implementation validation:
```bash
snyk code test src/upload/
snyk monitor
```

Include fixes for any Snyk findings in the implementation.
```

### VS Code Snyk Extension Configuration

```json
{
  "snyk.enableCodeQuality": true,
  "snyk.enableCodeSecurity": true,
  "snyk.features.openSourceSecurity": true,
  "snyk.features.codeSecurity": true,
  "snyk.features.infrastructureAsCode": true,
  "snyk.severity": {
    "critical": true,
    "high": true,
    "medium": true,
    "low": false
  },
  "snyk.scanOnSave": true,
  "snyk.ai.enabled": true,
  "snyk.ai.autofix": true
}
```

---

## Part 5: Comprehensive Security Prompt Examples

### OWASP Top 10 Focused Prompts

#### A01:2021 - Broken Access Control

**GitHub Copilot:**
```markdown
#codebase Create middleware that prevents OWASP A01:2021 vulnerabilities:

Implement these checks:
1. Verify user owns resource before operations
2. Check role permissions for each action  
3. Validate JWT hasn't been tampered with
4. Prevent parameter pollution
5. Block directory traversal attempts
6. Implement secure direct object references

Example attack vectors to prevent:
- GET /api/users/123/documents (accessing other user's documents)
- PUT /api/admin/settings (regular user accessing admin endpoints)
- GET /download?file=../../../../etc/passwd
- POST /api/orders with modified prices

Generate:
- Middleware functions
- Helper utilities
- Comprehensive test suite
- Documentation with security notes
```

#### A02:2021 - Cryptographic Failures

**Claude Code:**
```markdown
Role: Cryptography expert implementing OWASP A02:2021 compliant encryption

Implement secure data handling for PII:

Requirements:
1. Encryption at rest:
   - AES-256-GCM for sensitive fields
   - Unique IV per record
   - Key rotation support
   - HSM integration ready

2. Encryption in transit:
   - TLS 1.3 only
   - Certificate pinning
   - Perfect forward secrecy

3. Key management:
   - Derive keys with PBKDF2 (100,000 iterations)
   - Store in environment/KMS, never in code
   - Separate keys for different data types
   - Key versioning for rotation

4. Sensitive data handling:
   - No sensitive data in URLs
   - No caching of sensitive responses
   - Secure deletion (overwrite memory)
   - Redact from logs

Implementation with tests for:
- Encryption/decryption
- Key rotation
- Performance impact
- Error handling without leaking info
```

#### A03:2021 - Injection (Comprehensive)

**ChatGPT:**
```markdown
Create an API that's immune to all injection attacks:

Injection types to prevent:
1. SQL Injection
2. NoSQL Injection
3. Command Injection
4. LDAP Injection
5. XPath Injection
6. XML/XXE Injection
7. Template Injection
8. Expression Language Injection

For each endpoint:
- Input validation (allowlist)
- Parameterized queries/commands
- Output encoding
- Content-Type validation
- Schema validation
- Length limits

Example implementation for user search:
- SQL: Parameterized queries
- NoSQL: Sanitize operators ($gt, $ne)
- Command: No shell execution
- LDAP: Escape special characters
- XML: Disable external entities
- Template: Safe template engine config

Include tests with actual attack payloads:
- Polyglot payloads
- Encoding bypasses
- Comment injection
- Null byte injection
- Unicode attacks
```

#### A04:2021 - Insecure Design

**Claude Code:**
```markdown
Design and implement a secure password reset flow:

Threat model:
- Account enumeration
- Token prediction
- Token reuse
- Race conditions
- Social engineering

Security requirements:
1. Token generation:
   - Cryptographically secure (32 bytes)
   - Single-use
   - 15-minute expiration
   - Bound to user+timestamp

2. Rate limiting:
   - 3 requests per email per hour
   - 10 requests per IP per hour
   - Exponential backoff

3. Communication:
   - Generic success messages
   - No user enumeration
   - Secure email delivery
   - No token in URL after use

4. Validation:
   - Constant-time token comparison
   - Check expiration
   - Invalidate after use
   - Invalidate all on password change

5. Audit:
   - Log all attempts
   - Alert on suspicious patterns
   - Track token lifecycle

Generate complete implementation with tests.
```

#### A05:2021 - Security Misconfiguration

**GitHub Copilot:**
```markdown
Create security configuration validation:

Check for misconfigurations:
1. Default credentials
2. Unnecessary features enabled
3. Missing security headers
4. Verbose error messages
5. Open cloud storage
6. Unpatched dependencies
7. Permissive CORS
8. Debug mode in production

Implement:
- Startup configuration validator
- Security headers middleware:
  - Strict-Transport-Security
  - X-Frame-Options
  - X-Content-Type-Options
  - Content-Security-Policy
  - X-XSS-Protection
- Error handler that doesn't leak info
- CORS configuration with allowlist
- Dependency audit automation

Tests:
- Headers present and correct
- Error messages are generic
- No stack traces in production
- CORS blocks unauthorized origins
```

### Security Testing Prompt Templates

#### Template: Security Test Generation

**Claude Code:**
```markdown
Generate security tests for the authentication module:

Test categories:
1. Authentication bypass attempts
2. Session fixation
3. Concurrent session handling
4. Token manipulation
5. Brute force resistance
6. Account lockout bypass
7. Password reset vulnerabilities
8. Remember me vulnerabilities
9. OAuth/SAML attacks
10. JWT vulnerabilities

For each category, create tests that:
- Use realistic attack payloads
- Check both positive and negative cases
- Measure response times (timing attacks)
- Verify audit logs
- Check rate limiting

Output format:
- Jest/Mocha test suites
- Clear test descriptions
- Attack vector documentation
- Expected vs actual behavior
- Remediation notes for failures
```

---

## Part 6: Multi-Agent Security Workflows

### Security-Focused Agent Orchestra

#### Agent 1: Threat Modeler
```markdown
You are a threat modeling agent. Analyze this feature:

1. Identify assets (data, functions, resources)
2. List threat actors (external, internal, privileged)
3. Map attack vectors (STRIDE methodology)
4. Rank by likelihood and impact
5. Propose mitigations
6. Generate abuse cases

Output: Threat model document with risk matrix
```

#### Agent 2: Security Implementer
```markdown
You are implementing security controls from the threat model.

For each identified threat:
1. Select appropriate control type (preventive, detective, corrective)
2. Implement the control
3. Add monitoring/alerting
4. Document the implementation
5. Create bypass tests

Follow secure coding standards:
- OWASP guidelines
- CWE mitigation strategies
- Framework security best practices
```

#### Agent 3: Security Validator
```markdown
You are validating security implementations.

Tasks:
1. Run static analysis (CodeQL queries)
2. Execute dynamic tests (OWASP ZAP)
3. Check dependency vulnerabilities (Snyk)
4. Validate security configurations
5. Test attack scenarios
6. Generate security report

Mark code as: PASS, FAIL, or NEEDS_REVIEW
```

### Orchestrated Security Workflow Example

**Claude Code:**
```markdown
Orchestrate a security review for the payment module:

Agent sequence:
1. Threat Modeler: Identify PCI DSS relevant threats
2. Security Implementer: Add controls for each threat
3. Penetration Tester: Attempt to bypass controls
4. Security Validator: Run compliance checks
5. Documentation Agent: Generate security assessment

Coordination rules:
- Each agent must wait for previous agent's output
- Failed validation triggers re-implementation
- Maximum 3 iterations before escalation
- All changes must maintain existing tests

Output:
- Threat model
- Implemented controls with tests  
- Penetration test results
- Compliance checklist
- Security documentation
```

---

## Part 7: Production Security Checklist

### Pre-Deployment Security Gates

```markdown
## Security Checklist for AI-Generated Code

### Code Quality
- [ ] No hardcoded secrets (Snyk/GitGuardian scan)
- [ ] No unsafe deserialization
- [ ] No eval() or Function() with user input
- [ ] No SQL/NoSQL concatenation
- [ ] Parameterized queries only
- [ ] Input validation on all endpoints
- [ ] Output encoding for all user data
- [ ] Error messages don't leak information

### Dependencies
- [ ] No known vulnerabilities (Snyk/npm audit)
- [ ] Dependencies are from trusted sources
- [ ] Lock files are committed
- [ ] No unused dependencies
- [ ] License compliance verified

### Authentication/Authorization
- [ ] Strong password requirements
- [ ] Secure session management
- [ ] Rate limiting implemented
- [ ] Account lockout mechanism
- [ ] CSRF protection
- [ ] Proper authorization checks
- [ ] No privilege escalation paths

### Data Protection
- [ ] Encryption at rest for sensitive data
- [ ] TLS for data in transit
- [ ] PII is properly handled
- [ ] Secure key management
- [ ] Data retention policies
- [ ] Secure deletion implemented

### Infrastructure
- [ ] Security headers configured
- [ ] CORS properly restricted
- [ ] CSP policy implemented
- [ ] No debug mode in production
- [ ] Logging without sensitive data
- [ ] Monitoring and alerting setup

### Testing
- [ ] Security unit tests pass
- [ ] Integration tests include security cases
- [ ] Penetration testing completed
- [ ] CodeQL analysis clean
- [ ] Snyk scan passed
- [ ] Load testing for DoS resistance

### Documentation
- [ ] Security controls documented
- [ ] Threat model updated
- [ ] Incident response plan
- [ ] Security contact information
- [ ] Compliance requirements met
```

### Security-First PR Template

```markdown
## Pull Request: [Feature Name]

### Security Considerations
**Threat Model**: Link to threat model or inline description
**Attack Vectors Considered**:
- [ ] Injection attacks
- [ ] Authentication bypass
- [ ] Authorization flaws
- [ ] Data exposure
- [ ] XSS/CSRF
- [ ] DoS potential

### Security Controls Implemented
- Input validation: [Description]
- Authentication: [Method used]
- Authorization: [Checks implemented]
- Encryption: [What and how]
- Audit logging: [Events logged]

### Security Testing
- [ ] Unit tests for security controls
- [ ] Integration tests for auth flows
- [ ] Negative test cases (attack scenarios)
- [ ] CodeQL scan results: [PASS/FAIL]
- [ ] Snyk scan results: [PASS/FAIL]
- [ ] Manual security review completed

### Compliance
- [ ] OWASP Top 10 addressed
- [ ] GDPR/CCPA compliance (if applicable)
- [ ] PCI DSS compliance (if applicable)
- [ ] Internal security policies followed

### AI-Generated Code Declaration
**Percentage AI-generated**: ____%
**AI tool used**: [Claude Code/Copilot/ChatGPT]
**Additional review required**: [Yes/No]
**Security-focused prompts used**: [Yes/No]

### Rollback Plan
**How to rollback**: [Steps]
**Impact of rollback**: [Description]
**Monitoring after deployment**: [Metrics/Alerts]
```

---

## Part 8: Measurement and Continuous Improvement

### Security Metrics for AI-Assisted Development

```markdown
## Key Security Metrics

### Leading Indicators
1. Percentage of PRs with security tests
2. Time to remediate security findings
3. Number of security-focused prompts used
4. Security training completion rate
5. Pre-commit security scan adoption

### Lagging Indicators
1. Vulnerabilities per 1000 lines of code
2. Mean time to detect (MTTD) vulnerabilities
3. Mean time to remediate (MTTR)
4. Security incidents from AI-generated code
5. False positive rate in security scans

### AI-Specific Metrics
1. Security issues in AI vs human code
2. Prompt refinements needed for security
3. Autofix acceptance rate
4. Security test coverage for AI code
5. Time saved through security automation
```

### Continuous Improvement Process

```markdown
## Security Improvement Workflow

1. **Weekly Security Review**
   - Review all security findings
   - Identify patterns in AI-generated vulnerabilities
   - Update prompt templates
   - Refine CodeQL/Snyk rules

2. **Monthly Security Training**
   - New threat vectors
   - Tool updates (CodeQL, Snyk)
   - Prompt engineering for security
   - Case studies from incidents

3. **Quarterly Security Audit**
   - Full codebase security scan
   - Dependency audit
   - Penetration testing
   - Update threat models
   - Review and update security policies

4. **Incident Response**
   - Document security incidents
   - Root cause analysis
   - Update prompts to prevent recurrence
   - Share lessons learned
   - Update training materials
```

---

## Appendix A: Quick Reference Security Prompts

### SQL Injection Prevention
```
Implement [feature] with parameterized queries only. No string concatenation. Include tests for SQL injection attempts.
```

### XSS Prevention
```
Create [component] with proper output encoding. Use DOMPurify for user content. Test with XSS payloads.
```

### Authentication
```
Implement secure login with bcrypt (cost 12), rate limiting, account lockout, and session management. Include brute force tests.
```

### Authorization
```
Add role-based access control with deny-by-default, central checks, and audit logging. Test privilege escalation scenarios.
```

### Encryption
```
Implement AES-256-GCM encryption for [data type] with secure key management. Never hardcode keys. Include rotation support.
```

---

## Appendix B: Tool Configuration Files

### Complete VS Code Settings for Security-First AI Development

```json
{
  // Claude Code
  "claude.terminal.autoSave": true,
  "claude.security.enforceSecurePrompts": true,
  "claude.security.blockUnsafePatterns": true,
  
  // GitHub Copilot
  "github.copilot.enable": {
    "*": true,
    "yaml": true,
    "markdown": true
  },
  "github.copilot.advanced": {
    "securityMode": "strict",
    "authenticationRequired": true,
    "blockInsecurePatterns": true
  },
  
  // Security Extensions
  "snyk.enableCodeSecurity": true,
  "snyk.severity": {
    "critical": true,
    "high": true,
    "medium": true
  },
  "codeQL.runQueryOnChange": true,
  "semgrep.scan.onSave": true,
  
  // AI Security Settings
  "ai.security.scanBeforeCommit": true,
  "ai.security.requireSecurityReview": true,
  "ai.security.blockHighRiskPatterns": true,
  "ai.prompt.templates.security": "enabled",
  
  // Editor Security
  "files.exclude": {
    "**/.env": true,
    "**/*.key": true,
    "**/*.pem": true
  },
  "search.exclude": {
    "**/node_modules": true,
    "**/.git": true,
    "**/.env*": true
  }
}
```

---

## Conclusion

This comprehensive workshop guide provides a complete framework for transitioning teams to security-first AI-assisted development. By combining the power of Claude Code, GitHub Copilot, and ChatGPT with robust security tools like CodeQL and Snyk, teams can achieve both velocity and security. The key is treating AI as a powerful but potentially dangerous tool that requires careful prompting, continuous validation, and defense-in-depth security practices.

Remember: **AI accelerates both good and bad code equally. Security must be built into every prompt, every workflow, and every review cycle.**
