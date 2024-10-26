from app.main import workflow_app

result=workflow_app.invoke({'question':"What actors produce Adversarial Risks of OWASP LLM AI Checklist?" })

print("\n\n\n result:\n",result)