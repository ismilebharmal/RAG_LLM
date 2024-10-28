from app.main import workflow_app

# result=workflow_app.invoke({'question':"What actors produce Adversarial Risks of OWASP LLM AI Checklist?" })
# result=workflow_app.invoke({'question':"Who is the target audience of OWASP LLM AI Cybersecurity and Governance Checklist?" })
# result=workflow_app.invoke({'question':"What is the first risk on checklist??" })
result=workflow_app.invoke({'question':"Does the EU’s General Data Protection Regulation (GDPR) specifically address AI?" })



print("\n\n\n result:\n",result)