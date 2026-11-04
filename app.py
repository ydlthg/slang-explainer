"""
app.py —— 圈层黑话解释器（Streamlit 页面）
运行：streamlit run app.py
"""
import csv
import html
import os
import re
import time
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from slang_core import TERM_RE, explain

FEEDBACK_PATH = Path(os.getenv("SLANG_FEEDBACK_PATH", "data/feedback.csv"))
MAX_CALLS_PER_SESSION = 30            # 简单的防滥用：每个会话最多解释 30 次
EXAMPLES = [
    "因为这个我还上了wyy的热评",
    "这家店的yyds真的绝绝子，下次还来",
    "238这次回归不错，要是有kdy在就更好了",
]
ICON = {"高": "🟢", "中": "🟡", "低": "🔴"}

st.set_page_config(page_title="圈层黑话解释器", page_icon="🔎", layout="centered")
st.markdown(
    """<style>
    mark.slang{background:rgba(255,196,0,.35);color:inherit;border-radius:4px;padding:0 3px}
    mark.slang.uncertain,mark.slang.ai_guess{background:rgba(160,160,160,.35)}
    .orig{font-size:1.05rem;line-height:1.8}
    </style>""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def md(s) -> str:
    """转义 Markdown 特殊字符。词库候选是用户众包内容、解释来自 LLM，都视为不可信文本，
    避免被渲染成链接/图片。"""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~<])", r"\\\1", str(s or ""))


def highlight_html(sentence: str, result) -> str:
    """原句里高亮识别出的缩写。先转义再拼 HTML，用户输入不会被当作 HTML 执行。"""
    status = {t.term.lower(): t.status for t in result.terms}
    out, last = [], 0
    for m in TERM_RE.finditer(sentence):
        out.append(html.escape(sentence[last:m.start()]))
        tok = html.escape(m.group(0))
        s = status.get(m.group(0).lower())
        out.append(tok if s in (None, "not_slang") else f'<mark class="slang {s}">{tok}</mark>')
        last = m.end()
    out.append(html.escape(sentence[last:]))
    return '<div class="orig">' + "".join(out).replace("\n", "<br>") + "</div>"


def _csv_safe(v) -> str:
    v = str(v or "").replace("\n", " ")[:300]
    return "'" + v if v[:1] in ("=", "+", "-", "@") else v


def save_feedback(result, vote: str, comment: str) -> None:
    """只保存缩写、选择的含义、点赞/点踩和可选的纠错说明；不保存用户输入的原句。"""
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "vote", "terms_and_choices", "comment"])
        pairs = "; ".join(f"{t.term}={t.choice or '?'}({t.status})" for t in result.terms)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), vote, _csv_safe(pairs), _csv_safe(comment)])


# ----------------------------------------------------------------------
# 状态
# ----------------------------------------------------------------------
st.session_state.setdefault("sentence", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("calls", 0)
st.session_state.setdefault("fb_done", False)


def _fill(text: str):
    st.session_state.sentence = text


def run(sentence: str, ctx: str = ""):
    if not sentence.strip():
        st.warning("先粘贴一段话吧。")
        return
    if st.session_state.calls >= MAX_CALLS_PER_SESSION:
        st.error("本次使用次数已达上限，请稍后再试。")
        return
    st.session_state.calls += 1
    try:
        with st.spinner("正在查词库，并结合上下文推断含义…"):
            st.session_state.result = explain(sentence, ctx)
        st.session_state.fb_done = False
    except Exception:
        st.error("处理时出了点问题，请稍后重试。")


def _vote(v: str):
    if st.session_state.result is not None:
        save_feedback(st.session_state.result, v, st.session_state.get("fb_comment", ""))
        st.session_state.fb_done = True


# ----------------------------------------------------------------------
# 页面
# ----------------------------------------------------------------------
st.markdown('<h3 style="margin:0;">🔎 圈层黑话解释器</h3>', unsafe_allow_html=True)
st.caption(
    "粘贴一段含缩写/黑话的话，帮你读懂。词库来自开源项目 nbnhhsh（能不能好好说话），"
    "含义由 AI 结合整句上下文挑选，解释为 AI 生成，仅供参考。"
    "你输入的整句会发送给 AI 服务；只有抽取出的缩写会发给词库接口；反馈只保存缩写与含义，不保存原句。"
)

st.text_area("粘贴你看不懂的那段话", key="sentence", height=110, max_chars=500,
             placeholder="例如：这家店的yyds真的绝绝子，下次还来")
if st.button("解释", type="primary"):
    run(st.session_state.sentence)

st.caption("试试这些：")
cols = st.columns(len(EXAMPLES))
for c, ex in zip(cols, EXAMPLES):
    c.button(ex, on_click=_fill, args=(ex,), key=f"ex_{ex}", use_container_width=True)

res = st.session_state.result
if res is not None:
    st.divider()
    for w in res.warnings:
        st.warning(w)

    if not res.terms:
        st.info("没有识别到疑似缩写。这个工具主要解释字母/数字组成的缩写，比如 yyds、u1s1、jd。")
    else:
        if res.translation:
            st.markdown("**白话翻译**")
            st.success(md(res.translation))
        st.markdown("**原句**" + (f"　<small>语境：{html.escape(res.domain)}</small>" if res.domain else ""),
                    unsafe_allow_html=True)
        st.markdown(highlight_html(res.sentence, res), unsafe_allow_html=True)

        st.markdown("**逐词解释**")
        ignored = []
        for t in res.terms:
            if t.status == "not_slang":
                ignored.append(t.term)
                continue
            c1, c2, c3 = st.columns([1, 4, 1.4])
            c1.markdown(f"**{html.escape(t.term)}**")
            if t.status == "chosen":
                note = f"（{md(t.choice_note)}）" if t.choice_note else ""
                c2.markdown(md(t.choice) + note)
                if t.explanation:
                    c2.caption(md(t.explanation))
                if t.source == "inputting":
                    c2.caption("⚠️ 词库里只有输入联想，可靠性低")
                c3.markdown(f"{ICON.get(t.confidence, '🔴')} 置信度：{t.confidence}")
            elif t.status == "ai_guess":
                c2.markdown("AI 推测：" + md(t.choice))
                c2.caption("⚠️ 词库未收录，AI 推测，仅供参考")
                if t.explanation:
                    c2.caption(md(t.explanation))
                c3.markdown("🔴 置信度：低")
            else:   # uncertain
                c2.markdown("没有把握，请参考下面的词库候选" if t.others else "词库未收录，也没有可靠推测")
                if t.explanation:
                    c2.caption(md(t.explanation))
                c3.markdown("⚪ 不确定")
            if t.others:
                label = "词库候选" if t.status == "uncertain" else "其他可能含义"
                with st.expander(f"{label}（{len(t.others)}）"):
                    st.markdown(" · ".join(
                        md(c.text) + (f"（{md(c.note)}）" if c.note else "") for c in t.others))
        if ignored:
            st.caption("看起来不是黑话，已忽略：" + "、".join(html.escape(x) for x in ignored))

        st.divider()
        st.markdown("**结果不对？补充一句背景，让 AI 重新判断**")
        ctx = st.text_input("例如：这是在说韩娱", key="ctx", max_chars=100, label_visibility="collapsed",
                            placeholder="例如：这是在说韩娱")
        if st.button("带上背景重新解释"):
            run(res.sentence, ctx)
            st.rerun()

        st.markdown("**这次解释有帮助吗？**")
        if st.session_state.fb_done:
            st.success("谢谢反馈！")
        else:
            st.text_input("哪里不对？告诉我正确含义（可选）", key="fb_comment", max_chars=100)
            b1, b2, _ = st.columns([1, 1, 4])
            b1.button("👍 有帮助", on_click=_vote, args=("up",))
            b2.button("👎 不准确", on_click=_vote, args=("down",))
