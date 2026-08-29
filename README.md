# 🐸 CodeFrog AI

### Your AI Software Engineer for GitHub

> **Jump into your GitHub codebase. Build, fix, test, and ship faster.**

CodeFrog AI is an AI-powered developer platform that connects with GitHub repositories and acts like an intelligent software engineer.

Instead of only answering coding questions, CodeFrog understands an entire codebase, analyzes files and dependencies, identifies bugs and security issues, generates code changes, runs tests, and creates Pull Requests for developer approval.

---

## 🚀 What is CodeFrog AI?

Modern software projects can contain hundreds or thousands of files. Understanding where a problem exists, which files are affected, and how to safely fix it can take a lot of time.

CodeFrog AI is designed to make this process faster.

A developer can connect a GitHub repository and simply tell CodeFrog what they want:

```text
"Find all APIs without authentication and fix them."
```

CodeFrog can then:

```text
Understand the request
        ↓
Analyze the repository
        ↓
Find relevant files
        ↓
Identify the problem
        ↓
Create a fix plan
        ↓
Generate code changes
        ↓
Show the developer a diff
        ↓
Run tests
        ↓
Run security checks
        ↓
Ask for human approval
        ↓
Create a Git branch
        ↓
Commit the changes
        ↓
Create a Pull Request
```

The developer remains in control of the final changes.

---

# 🎯 Vision

CodeFrog AI aims to become an **AI Software Engineer for GitHub repositories**.

The goal is not to build another coding chatbot.

The goal is to build an agent that can:

* Understand a complete codebase
* Search and analyze source code
* Detect bugs
* Find security vulnerabilities
* Suggest and generate fixes
* Generate tests
* Run tests
* Review code
* Modify files
* Create branches
* Commit changes
* Create Pull Requests

### Core idea

> **CodeFrog doesn't just tell you what to change. It can understand the repository, prepare the change, test it, and submit it for your review.**

---

# 🐸 Why "CodeFrog"?

The name represents the core idea behind the product.

**Frog → Jump**

CodeFrog can "jump into" a developer's GitHub repository and work with the codebase.

**Code → Software**

The platform understands and works with source code, APIs, dependencies, tests, and project architecture.

**AI → Intelligent Engineer**

AI analyzes the repository and performs development tasks using specialized tools.

### Brand statement

> **CodeFrog AI — Jump into your codebase. Ship faster.**

---

# ⭐ Core Features

## 1. 🔗 GitHub Integration

Connect your GitHub account and select the repository you want CodeFrog to work with.

### Planned capabilities

* GitHub OAuth
* Repository selection
* Repository file access
* GitHub API integration
* GitHub Webhooks
* Branch management
* Commit creation
* Pull Request creation

---

# 2. 🤖 AI Developer Chat

CodeFrog provides an AI interface where developers can ask questions about their actual codebase.

Example questions:

```text
How does authentication work?
```

```text
Where is task creation implemented?
```

```text
Which API handles user registration?
```

```text
Explain this controller.
```

```text
Why is this function slow?
```

```text
How does data move from frontend to backend?
```

The AI answers using relevant code from the connected repository rather than relying only on generic programming knowledge.

---

# 3. 🧠 Codebase Understanding with RAG

Large repositories can contain hundreds or thousands of files.

Sending the entire repository to an LLM for every request would be inefficient.

CodeFrog uses a Retrieval-Augmented Generation architecture.

### Repository indexing flow

```text
GitHub Repository
        ↓
Repository Parser
        ↓
Code Files
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector Database
        ↓
Retriever
        ↓
Relevant Code
        ↓
LLM
        ↓
AI Response
```

For example, if a developer asks:

```text
"How does authentication work?"
```

CodeFrog can retrieve relevant files such as:

```text
Login.jsx
authController.js
authService.js
authMiddleware.js
User.js
```

Instead of sending the entire repository to the model.

---

# 4. 🛠️ AI Software Engineer Agent

## The Core Feature

The AI Agent is the heart of CodeFrog AI.

A developer can give the agent a high-level software engineering task.

For example:

```text
Find all APIs that don't have authentication and fix them.
```

The agent can break this request into multiple steps.

```text
User Request
      ↓
Understand Task
      ↓
Analyze Repository
      ↓
Find Relevant Files
      ↓
Identify Problem
      ↓
Create Fix Plan
      ↓
Generate Changes
      ↓
Run Tests
      ↓
Security Check
      ↓
Show Diff
      ↓
Human Approval
      ↓
Create Branch
      ↓
Commit Changes
      ↓
Create Pull Request
```

This makes CodeFrog an **agentic software development system** rather than a simple chatbot.

---

# 5. 🔍 Security Analysis

CodeFrog can analyze repositories for common security problems.

Example requests:

```text
Find IDOR vulnerabilities.
```

```text
Find APIs without authentication.
```

```text
Find hardcoded secrets.
```

```text
Check insecure CORS configuration.
```

```text
Find missing authorization checks.
```

Example result:

```text
🔍 Security Analysis

Found 5 potentially unprotected endpoints:

1. GET /api/users
2. GET /api/tasks
3. POST /api/submissions
4. GET /api/reports
5. DELETE /api/tasks/:id
```

The agent can explain why each issue may be dangerous and suggest an appropriate fix.

---

# 6. 🐛 AI Bug Finder

Developers can provide an error or describe unexpected behavior.

Example:

```text
TypeError: Cannot read properties of undefined
```

CodeFrog can search the repository and identify the likely source.

Example:

```text
Possible source:

userService.js

getUser() can return null, but the caller
directly accesses user.name.

Suggested fix:
Add null handling before accessing user.name.
```

The agent can then prepare a code change for review.

---

# 7. 🧪 AI Test Generator

CodeFrog can generate tests for existing functions and features.

For example:

```text
Generate tests for createUser().
```

Possible test cases:

```text
✓ Valid user
✓ Duplicate email
✓ Missing email
✓ Invalid password
✓ Unauthorized request
✓ Database error
```

The system can generate tests using frameworks such as:

* Jest
* Vitest
* Other project-compatible testing frameworks

---

# 8. 🔎 AI Code Review

CodeFrog can analyze code changes and provide feedback.

Example:

```text
Code Quality: 85/100
Security:      72/100
Performance:   90/100
Testing:       65/100
```

Possible findings:

```text
⚠ Authorization check missing
⚠ Possible IDOR vulnerability
⚠ Database query can be optimized
⚠ Missing test coverage
```

Code review can be integrated into Pull Request workflows.

---

# 9. 👀 Code Diff Viewer

CodeFrog should never silently change a repository.

Before creating a Pull Request, the developer should be able to see exactly what the AI wants to change.

Example:

```diff
- router.get('/tasks', getTasks);

+ router.get(
+   '/tasks',
+   authenticateUser,
+   getTasks
+ );
```

Another example:

```diff
- router.delete('/tasks/:id', deleteTask);

+ router.delete(
+   '/tasks/:id',
+   authenticateUser,
+   authorizeAdmin,
+   deleteTask
+ );
```

The developer can review the proposed changes before approving them.

---

# 10. 🧪 Automated Test Execution

After generating changes, CodeFrog can run the project's tests.

Example:

```text
Running tests...

✓ Authentication test
✓ Unauthorized request test
✓ Authorized request test
✓ Admin authorization test
✓ Existing task test

32 passed
0 failed
```

If tests fail, the agent can analyze the failures and propose another fix.

---

# 11. 🔐 Human-in-the-Loop Security

CodeFrog follows an important principle:

> **AI should not directly modify the production or main branch without human approval.**

The intended workflow is:

```text
AI
 ↓
Analyze
 ↓
Plan
 ↓
Generate Changes
 ↓
Show Diff
 ↓
Run Tests
 ↓
Security Scan
 ↓
👤 User Approval
 ↓
Create Branch
 ↓
Commit
 ↓
Create Pull Request
```

The developer remains responsible for approving the final changes.

---

# 12. 🌳 Automatic GitHub Workflow

After approval, CodeFrog can perform Git operations.

Example:

### Branch

```text
ai/fix-api-authentication
```

### Commit

```text
fix: secure unprotected API endpoints
```

### Pull Request

```text
AI Security Fixes

5 API endpoints were missing
authentication/authorization checks.

Files changed: 6

Tests: 32 passed

Security checks: Passed
```

The developer can then review and merge the Pull Request.

---

# 🧰 AI Agent Tools

The agent will use specialized tools to interact with the repository.

Planned tools include:

```text
search_code()
read_file()
list_files()

analyze_routes()
analyze_dependencies()

generate_patch()

run_tests()
run_security_scan()

create_branch()
commit_changes()
create_pull_request()
```

The AI decides which tools are required for each task.

Example:

```text
User:
"Find APIs without authentication."

        ↓

analyze_routes()

        ↓

read_file()

        ↓

analyze authentication middleware

        ↓

identify vulnerable endpoints
```

For a fix:

```text
generate_patch()
        ↓
run_tests()
        ↓
run_security_scan()
        ↓
show_diff()
        ↓
user approval
        ↓
create_branch()
        ↓
commit_changes()
        ↓
create_pull_request()
```

---

# 💡 Example Agent Tasks

## Security

```text
Find IDOR vulnerabilities.
```

```text
Find APIs without authentication.
```

```text
Find hardcoded secrets.
```

```text
Check insecure CORS configuration.
```

## Bug Fixing

```text
Find why login is failing.
```

```text
Fix this TypeError.
```

```text
Find APIs returning 500 errors.
```

## Code Quality

```text
Find duplicate code.
```

```text
Refactor this controller.
```

```text
Improve this database query.
```

## Testing

```text
Generate tests for authentication.
```

```text
Find untested functions.
```

## Documentation

```text
Generate API documentation.
```

```text
Update README according to the current project.
```

## Performance

```text
Find slow database queries.
```

```text
Find unnecessary API calls.
```

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │       Developer     │
                         └──────────┬──────────┘
                                    │
                                    ↓
                         ┌─────────────────────┐
                         │   CodeFrog Web App  │
                         └──────────┬──────────┘
                                    │
                                    ↓
                         ┌─────────────────────┐
                         │    CodeFrog API     │
                         └──────────┬──────────┘
                                    │
                                    ↓
                         ┌─────────────────────┐
                         │     AI Agent        │
                         └──────┬───────┬──────┘
                                │       │
                 ┌──────────────┘       └──────────────┐
                 ↓                                     ↓
        ┌─────────────────┐                   ┌─────────────────┐
        │ Repository Tools│                   │     LLM         │
        └────────┬────────┘                   └─────────────────┘
                 │
                 ↓
        ┌─────────────────┐
        │ GitHub API      │
        └────────┬────────┘
                 │
                 ↓
        ┌─────────────────┐
        │ GitHub Repository│
        └─────────────────┘
```

---

# 🧠 RAG Architecture

```text
GitHub Repository
        │
        ↓
Repository Parser
        │
        ↓
Code Files
        │
        ↓
Chunking
        │
        ↓
Embeddings
        │
        ↓
Vector Database
        │
        ↓
Retriever
        │
        ↓
Relevant Code
        │
        ↓
LLM
        │
        ↓
Agent / Response
```

---

# 💻 Technology Stack

## Frontend

* React / Next.js
* TypeScript
* Tailwind CSS

## Backend

* Python
* FastAPI
* Uvicorn
* Pydantic
* SQLAlchemy

## Database

* PostgreSQL
* Redis
* pgvector

## AI

* LLM API
* Embeddings
* RAG
* Vector Search
* AI Agent / Tool Calling

## GitHub Integration

* GitHub OAuth
* GitHub API
* GitHub Webhooks
* Git branches
* Git commits
* Pull Requests

## DevOps

* Docker
* GitHub Actions
* Cloud deployment

---

# 📁 Project Structure

The initial repository is planned as a monorepo:

```text
codefrog/
│
├── frontend/
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   ├── services/
│   └── ...
│
├── backend/
│   ├── controllers/
│   ├── routes/
│   ├── models/
│   ├── services/
│   ├── middleware/
│   └── ...
│
├── agent/
│   ├── tools/
│   ├── prompts/
│   ├── analyzers/
│   └── ...
│
├── README.md
├── .gitignore
└── docker-compose.yml
```

---

# 🖥️ Product Interface

The CodeFrog dashboard will provide access to:

```text
Dashboard
Repositories
AI Agent
Code Review
Bug Finder
Test Generator
Security
Pull Requests
Settings
```

### AI Agent interface

```text
┌──────────────────────────────────────────────┐
│ 🐸 CodeFrog AI                               │
│                                              │
│ What should I work on?                       │
│                                              │
│ Find security vulnerabilities and fix them   │
│ without changing existing APIs.              │
│                                              │
│                         [ Run Agent → ]       │
└──────────────────────────────────────────────┘
```

---

# 🔄 Complete User Workflow

```text
1. Create CodeFrog account
             ↓
2. Connect GitHub
             ↓
3. Select repository
             ↓
4. CodeFrog indexes repository
             ↓
5. Developer gives AI a task
             ↓
6. Agent analyzes repository
             ↓
7. Agent creates a plan
             ↓
8. Developer reviews plan
             ↓
9. Agent generates code changes
             ↓
10. Developer reviews diff
             ↓
11. Agent runs tests
             ↓
12. Agent runs security checks
             ↓
13. Developer approves
             ↓
14. Agent creates branch
             ↓
15. Agent commits changes
             ↓
16. Agent creates Pull Request
             ↓
17. Developer reviews PR
             ↓
18. Developer merges changes
```

---

# 🛡️ Security Principles

CodeFrog is designed with developer control and repository security in mind.

### Core principles

* Never expose GitHub access tokens to the frontend
* Use OAuth for GitHub authentication
* Use least-privilege GitHub permissions
* Never automatically merge AI-generated Pull Requests
* Require human approval before repository changes
* Isolate code execution environments
* Validate AI-generated patches
* Run tests before creating Pull Requests
* Run security checks on generated changes
* Protect sensitive environment variables
* Never expose repository secrets to the LLM unnecessarily

---

# 🚀 Development Roadmap

## Phase 1 — Foundation

* [ ] Create CodeFrog GitHub organization
* [ ] Create CodeFrog Web repository
* [ ] Set up frontend
* [ ] Set up backend
* [ ] Set up database
* [ ] Create initial dashboard

## Phase 2 — GitHub Integration

* [ ] GitHub OAuth
* [ ] GitHub account connection
* [ ] Repository listing
* [ ] Repository selection
* [ ] Repository file access
* [ ] GitHub API integration

## Phase 3 — Codebase Intelligence

* [ ] Repository parser
* [ ] File indexing
* [ ] Code chunking
* [ ] Embeddings
* [ ] pgvector integration
* [ ] RAG pipeline
* [ ] AI code chat

## Phase 4 — AI Agent

* [ ] Agent architecture
* [ ] Tool calling
* [ ] Code search tool
* [ ] File reading tool
* [ ] Route analyzer
* [ ] Dependency analyzer
* [ ] Patch generator
* [ ] Test runner
* [ ] Security scanner

## Phase 5 — Code Modification

* [ ] Generate code changes
* [ ] Generate patches
* [ ] Diff viewer
* [ ] User approval
* [ ] Change validation

## Phase 6 — GitHub Automation

* [ ] Create branch
* [ ] Commit changes
* [ ] Create Pull Request
* [ ] PR status tracking
* [ ] GitHub Webhooks

## Phase 7 — Advanced Features

* [ ] AI code review
* [ ] AI bug finder
* [ ] AI test generator
* [ ] Security analysis
* [ ] Performance analysis
* [ ] Documentation generation
* [ ] Agent task history
* [ ] Analytics

---

# 🎯 Example

Developer:

```text
Find all APIs that don't have authentication and fix them.
```

CodeFrog:

```text
🔍 Analyzing repository...

Repository: talent-portal
Files analyzed: 247

Found 5 potentially unprotected endpoints:

1. GET /api/users
2. GET /api/tasks
3. POST /api/submissions
4. GET /api/reports
5. DELETE /api/tasks/:id
```

CodeFrog creates a plan:

```text
AI PLAN

1. Add authentication middleware
2. Add authorization checks
3. Update affected routes
4. Update affected tests

Estimated files changed: 6
```

Developer:

```text
[ Generate Changes ]
```

CodeFrog:

```text
Changes Generated

6 files modified

[ View Diff ]
```

Developer:

```text
[ Run Tests ]
```

CodeFrog:

```text
✓ 32 tests passed
✓ Security checks passed
```

Developer:

```text
[ Approve Changes ]
```

CodeFrog:

```text
Branch:
ai/fix-api-authentication

Commit:
fix: secure unprotected API endpoints

Pull Request:
#128
```

This is the core CodeFrog experience.

---

# 🏆 What Makes CodeFrog Different?

CodeFrog is not designed to be another AI chatbot.

A traditional chatbot:

```text
Developer
   ↓
Question
   ↓
AI
   ↓
Answer
```

CodeFrog:

```text
Developer
   ↓
Task
   ↓
AI Agent
   ↓
Understand Repository
   ↓
Search Code
   ↓
Analyze
   ↓
Plan
   ↓
Generate Changes
   ↓
Run Tests
   ↓
Security Check
   ↓
Human Approval
   ↓
GitHub Branch
   ↓
Commit
   ↓
Pull Request
```

### The goal

> **Understand → Analyze → Modify → Test → Review → Ship**

---

# 📌 Project Status

🚧 **Currently in development**

CodeFrog AI is being developed as an AI + Full Stack engineering project focused on GitHub repository understanding and autonomous software development workflows.

---

# 🗺️ Future Vision

CodeFrog AI can eventually evolve into a complete AI engineering platform capable of helping developers with:

* Repository maintenance
* Security remediation
* Bug fixing
* Refactoring
* Testing
* Documentation
* Performance optimization
* Code review
* Dependency upgrades
* Pull Request automation
* Continuous repository analysis

The long-term vision is simple:

> **Give CodeFrog a software engineering task and let it do the engineering work — while keeping the developer in control.**

---

# 👨‍💻 Author

**Shishir Mahato**

Computer Science Engineering — Data Science

---

# 📄 License

License information will be added as the project develops.

---

<p align="center">

### 🐸 CodeFrog AI

**Jump into your codebase. Ship faster.**

</p>

---

## Local backend setup

The backend reads configuration from environment variables and a repository-root `.env` file. Never commit `.env` or place real credentials in `.env.example`.

1. Copy the safe template and replace every `YOUR_...` placeholder with local-only values:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Start PostgreSQL. Docker Compose reads `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` from `.env`:

   ```powershell
   docker compose up -d postgres
   ```

3. Run the API from the repository root:

   ```powershell
   uvicorn main:app --app-dir backend --reload
   ```

`DATABASE_URL` and `AUTH_SECRET_KEY` are required. Generate a unique local `AUTH_SECRET_KEY` of at least 32 characters; it signs short-lived local authentication tokens and must never be committed. `APP_NAME`, `APP_ENV` (`development`, `test`, or `production`), and `LOG_LEVEL` are optional. Use a PostgreSQL psycopg URL, such as `postgresql+psycopg://USER:PASSWORD@localhost:5432/DATABASE`.

Local account passwords must be at least 12 characters. They are stored only as Argon2 hashes; registration and public-user responses never expose password hashes.

Run the backend tests with:

```powershell
python -m pytest backend/tests -q
```



