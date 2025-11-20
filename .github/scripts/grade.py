#!/usr/bin/env python3
"""
AI-Powered Assignment Grading Script
AI驱动的作业评分脚本

Supports multiple AI providers:
- OpenAI (GPT-4, GPT-3.5)
- 智谱AI (GLM-4.6 with thinking)
- Can be extended to other providers

支持多个AI提供商：
- OpenAI (GPT-4, GPT-3.5)
- 智谱AI (GLM-4.6 带思维链)
- 可扩展到其他提供商
"""

import os
import json
import requests
from pathlib import Path

# Detect which AI provider to use based on available API key
AI_PROVIDER = os.environ.get("AI_PROVIDER", "zhipu")  # Default to zhipu

if AI_PROVIDER == "openai" and os.environ.get("OPENAI_API_KEY"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    MODEL = "gpt-4o"
    print("🤖 Using OpenAI GPT-4o")
    
elif AI_PROVIDER == "zhipu" or os.environ.get("ZHIPU_API_KEY"):
    # Use requests for direct API call to support GLM-4.6 features
    ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")
    MODEL = "glm-4.6"  # GLM-4.6 with thinking capability
    print("🤖 Using 智谱AI GLM-4.6 (with thinking)")
    
else:
    raise ValueError("No AI API key found. Please set OPENAI_API_KEY or ZHIPU_API_KEY")

# Grading rubric
RUBRIC = """
# Day 1 Morning Introduction - Grading Rubric

## File 1: my-maker-profile.md
Required sections:
1. About Me - Name and at least 2 hobbies/interests
2. Why I'm Here - At least 2-3 sentences explaining motivation
3. Project Idea - Clear description (3-5 sentences)
4. Project Reasoning - Why they want to make it
5. Skills Assessment - At least 3 skills rated
6. Maker Identity - One principle chosen and explained
7. 6-Day Goals - At least 3 specific goals

## File 2: challenge-reflection.md
Required sections:
1. Challenge Results - Team info and final height
2. Team Members - All members listed
3. What Worked - 2-3 sentences on successful strategies
4. What Failed - Honest reflection on failures
5. Iterations - Description of design changes
6. Teamwork Lessons - Thoughtful collaboration insights
7. Engineering Thinking - Connection to engineering principles
8. Maker Mindset - Connection to at least 3 Maker principles
9. Personal Reflection - Honest sharing of feelings

## Evaluation Criteria:
- **Completeness**: All required sections filled in (not just template)
- **Thoughtfulness**: Answers show genuine reflection, not rushed
- **Specificity**: Concrete details, not vague statements
- **Language**: Can be English or Chinese, both acceptable
- **Length**: Appropriate depth for each section
"""

GRADING_PROMPT = """
You are an experienced Maker educator evaluating a student's Day 1 morning introduction assignment.

你是一位经验丰富的创客教育者，正在评估学生第1天上午的自我介绍作业。

**Your task | 你的任务:**
1. Read the student's submissions carefully | 仔细阅读学生的提交内容
2. Check against the rubric requirements | 对照评分标准检查
3. Provide constructive, encouraging feedback in both English and Chinese | 提供建设性的、鼓励性的中英双语反馈
4. Focus on completion and thoughtfulness, not perfection | 关注完成度和深思熟虑，而非完美
5. Highlight what the student did well | 强调学生做得好的地方
6. Suggest specific improvements if needed | 如果需要，提出具体改进建议

**Tone | 语气:** Warm, encouraging, constructive. Remember this is Day 1!
温暖、鼓励、建设性。记住这是第1天！

**Output format | 输出格式:** Generate a Markdown feedback report with:
- Overall completion status (✅ Complete / ⚠️ Needs revision)
- Section-by-section checklist
- Specific praise (what they did well)
- Gentle suggestions for improvement (if any)
- Encouraging closing message

生成Markdown反馈报告，包含：
- 总体完成状态（✅ 完成 / ⚠️ 需要修订）
- 逐部分检查清单
- 具体表扬（做得好的地方）
- 温和的改进建议（如有）
- 鼓励性的结束语

Here is the rubric | 评分标准:
{rubric}

Here are the student's submissions | 学生提交的内容:
{submissions}
"""

def read_file_safe(filepath):
    """Safely read a file, return content or error message"""
    try:
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"❌ File not found: {filepath}"
    except Exception as e:
        return f"❌ Error reading {filepath}: {str(e)}"

def call_ai_api(prompt, system_message):
    """Call AI API with provider-specific logic"""
    if AI_PROVIDER == "openai":
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    elif AI_PROVIDER == "zhipu":
        # Use direct API call for GLM-4.6 with thinking
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {ZHIPU_API_KEY}"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "thinking": {
                "type": "enabled"  # Enable thinking for better reasoning
            },
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    else:
        raise ValueError(f"Unknown AI provider: {AI_PROVIDER}")

def main():
    print("🤖 Starting AI grading process...")
    
    # Read student submissions
    maker_profile = read_file_safe("my-maker-profile.md")
    challenge_reflection = read_file_safe("challenge-reflection.md")
    
    submissions = f"""
## Student Submission 1: my-maker-profile.md
```markdown
{maker_profile}
```

## Student Submission 2: challenge-reflection.md
```markdown
{challenge_reflection}
```
"""
    
    print("📝 Files read successfully")
    print(f"   - my-maker-profile.md: {len(maker_profile)} characters")
    print(f"   - challenge-reflection.md: {len(challenge_reflection)} characters")
    
    # Call AI API
    print(f"🧠 Calling {AI_PROVIDER.upper()} AI model: {MODEL}...")
    
    try:
        system_message = "You are a warm, encouraging Maker educator providing feedback on student assignments. You are bilingual in English and Chinese. 你是一位温暖、鼓励学生的创客教育者，正在为学生作业提供反馈。你精通中英双语。"
        
        prompt = GRADING_PROMPT.format(
            rubric=RUBRIC,
            submissions=submissions
        )
        
        feedback = call_ai_api(prompt, system_message)
        
        print("✅ AI grading completed")
        
        # Add header and footer
        full_feedback = f"""# 📝 AI Grading Feedback | AI评分反馈

> **Assignment**: Day 1 Morning - Maker Introduction  
> **作业**: 第1天上午 - Maker自我介绍  
> **Graded by**: AI Teaching Assistant ({MODEL})  
> **评分者**: AI助教 ({MODEL})  
> **Provider**: {AI_PROVIDER.upper()}  
> **提供商**: {AI_PROVIDER.upper()}  
> **Date**: {os.popen('date').read().strip()}  
> **日期**: {os.popen('date').read().strip()}

---

{feedback}

---

## 📌 Next Steps | 下一步

If your assignment is marked as **✅ Complete**, great work! You're all set.

如果你的作业标记为 **✅ 完成**，做得好！你已经完成了。

If it's marked as **⚠️ Needs revision**, please address the suggestions above and push your changes. The AI will automatically re-grade your work.

如果标记为 **⚠️ 需要修订**，请根据上面的建议进行修改并推送更改。AI 会自动重新评分。

---

## 💬 Questions? | 有疑问？

If you have questions about this feedback:
1. Review the [rubric.md](./rubric.md) for detailed criteria
2. Check the assignment [README.md](./README.md)
3. Contact your instructor for clarification

如果对反馈有疑问：
1. 查看 [rubric.md](./rubric.md) 了解详细标准
2. 查看作业 [README.md](./README.md)
3. 联系讲师寻求澄清

---

*This is an automated AI grading. Your instructor may provide additional feedback or override this assessment.*  
*这是 AI 自动评分。讲师可能会提供额外反馈或推翻此评估。*
"""
        
        # Save feedback
        with open("feedback.md", "w", encoding="utf-8") as f:
            f.write(full_feedback)
        
        print("💾 Feedback saved to feedback.md")
        
        # Also save as JSON for potential further processing
        feedback_data = {
            "timestamp": os.popen('date').read().strip(),
            "provider": AI_PROVIDER,
            "model": MODEL,
            "feedback": feedback,
            "files_checked": {
                "my-maker-profile.md": len(maker_profile) > 100,
                "challenge-reflection.md": len(challenge_reflection) > 100
            }
        }
        
        with open("grading-result.json", "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=2, ensure_ascii=False)
        
        print("✨ Grading complete!")
        
    except Exception as e:
        print(f"❌ Error during AI grading: {str(e)}")
        
        # Create error feedback
        error_feedback = f"""# ❌ AI Grading Error | AI评分错误

An error occurred while grading your assignment:

评分过程中发生错误：

```
{str(e)}
```

**Provider**: {AI_PROVIDER.upper()}
**Model**: {MODEL}

Please contact your instructor for manual grading.

请联系讲师进行手动评分。

**Possible reasons | 可能的原因:**
- API key not configured correctly | API密钥配置不正确
- API rate limit exceeded | API速率限制超出
- Network connectivity issue | 网络连接问题
- API service temporarily unavailable | API服务暂时不可用
"""
        with open("feedback.md", "w", encoding="utf-8") as f:
            f.write(error_feedback)
        
        raise

if __name__ == "__main__":
    main()
