"""邮件发送（邮箱确认 magic link 用）。

配置来源（优先级从高到低）：
1. 控制台「设置 → 邮箱发信」保存的配置（存 data/relay.db 的 meta 表）
2. 环境变量 RELAY_SMTP_* / RELAY_MAIL_MODE
3. 默认：没配 SMTP 服务器就退日志模式——确认链接写进服务器日志
   （[mail/log] 前缀），自托管/内网部署不用配 SMTP 也能闭环。

发信失败不阻塞注册：注册照常吃下，账号保持 pending，可在登录页重发。
"""
from __future__ import annotations

import logging
import os
import smtplib

log = logging.getLogger("relay.mail")

# meta 表里的配置键（控制台「邮箱发信」设置页写入）
META = {
    "mode": "mail_mode",
    "host": "mail_smtp_host",
    "port": "mail_smtp_port",
    "user": "mail_smtp_user",
    "pass": "mail_smtp_pass",
    "from": "mail_smtp_from",
    "starttls": "mail_smtp_starttls",
}


def _m(store, field: str) -> str:
    return (store.get_meta(META[field]) or "").strip()


def mail_settings(store) -> dict:
    """当前生效的 SMTP 配置（控制台设置 > 环境变量 > 默认）。"""
    host = _m(store, "host") or os.getenv("RELAY_SMTP_HOST", "").strip()
    port_raw = _m(store, "port") or os.getenv("RELAY_SMTP_PORT", "").strip()
    try:
        port = int(port_raw) if port_raw else 587
    except ValueError:
        port = 587
    user = _m(store, "user") or os.getenv("RELAY_SMTP_USER", "").strip()
    pwd = _m(store, "pass") or os.getenv("RELAY_SMTP_PASS", "")
    from_addr = _m(store, "from") or os.getenv("RELAY_SMTP_FROM", "").strip() or user
    st_raw = _m(store, "starttls") or os.getenv("RELAY_SMTP_STARTTLS", "1").strip()
    starttls = st_raw != "0"
    mode = _m(store, "mode").lower()
    if mode not in ("smtp", "log"):
        if os.getenv("RELAY_MAIL_MODE", "").strip().lower() == "log":
            mode = "log"
        else:
            mode = "smtp" if host else "log"
    if mode == "smtp" and not host:
        mode = "log"   # 选了 smtp 但没有服务器 → 实际只能落日志
    return {"mode": mode, "host": host, "port": port, "user": user,
            "pass": pwd, "from": from_addr, "starttls": starttls}


def public_mail_settings(store) -> dict:
    """给控制台「邮箱发信」页展示的配置（密码打码，不返回明文）。"""
    d = mail_settings(store)
    raw = d.pop("pass", "")
    d["has_pass"] = bool(raw)
    d["pass_masked"] = ("••••" + raw[-4:]) if len(raw) >= 4 else ("••••" if raw else "")
    return d


def save_mail_settings(store, mode: str, host: str, port, user: str,
                      password: str, from_addr: str, starttls: bool) -> None:
    """保存控制台设置（meta 表）。密码留空 = 保持旧值（UI 只能看到打码）。"""
    store.set_meta(META["mode"], mode)
    store.set_meta(META["host"], str(host or "").strip())
    store.set_meta(META["port"], "" if port in (None, "") else str(port))
    store.set_meta(META["user"], str(user or "").strip())
    if password:   # 空密码不落库，沿用已有的
        store.set_meta(META["pass"], str(password))
    store.set_meta(META["from"], str(from_addr or "").strip())
    store.set_meta(META["starttls"], "1" if starttls else "0")


def _send(store, to_email: str, subject: str, body: str, html: str = "") -> dict:
    """发信。html 非空时走 multipart/alternative：HTML 版（链接可点）+ 纯文本兜底。"""
    d = mail_settings(store)
    if d["mode"] == "log":
        log.warning("[mail/log] %s → %s：%s", subject, to_email, body)
        return {"mode": "log", "ok": True, "error": ""}
    try:
        from_addr = d["from"] or to_email
        from email.message import EmailMessage
        msg = EmailMessage()
        msg.set_content(body)
        if html:
            msg.add_alternative(html, subtype="html")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_email
        with smtplib.SMTP(d["host"], d["port"], timeout=15) as s:
            if d["starttls"]:
                s.starttls()
            if d["user"]:
                s.login(d["user"], d["pass"])
            s.send_message(msg)
        return {"mode": "smtp", "ok": True, "error": ""}
    except Exception as e:  # 发信失败不抛给调用方：注册仍成立，可重发
        log.error("[mail/smtp] 发信失败 to=%s: %s", to_email, e)
        return {"mode": "smtp", "ok": False, "error": str(e)}


def send_verification_email(store, to_email: str, link: str,
                            from_name: str = "LLM Relay") -> dict:
    """发「邮箱确认」邮件。返回 {"mode": "smtp"|"log", "ok": bool, "error": str}。

    HTML 版（链接可点，按钮样式）+ 纯文本兜底（老客户端直接显示链接文本）。"""
    import html as _h
    subject = f"{from_name}：请确认你的邮箱"
    body = (f"欢迎使用 {from_name}。\n\n"
            f"请点击下面的链接完成邮箱确认（链接 30 分钟内有效）：\n\n"
            f"{link}\n\n"
            f"如果这不是你本人操作的注册，请直接忽略这封邮件。\n")
    hl = _h.escape(link, quote=True)
    html = (
        '<div style="background:#f4f6f8;padding:24px;font-family:-apple-system,'
        '"Segoe UI","Microsoft YaHei",sans-serif">'
        '<div style="max-width:520px;margin:0 auto;background:#fff;border-radius:12px;'
        'padding:32px 28px;border:1px solid #e5e9f0">'
        f'<h1 style="font-size:18px;margin:0 0 12px;color:#1a2233">{_h.escape(from_name)}：请确认你的邮箱</h1>'
        '<p style="font-size:14px;line-height:1.8;color:#3d4759;margin:0 0 18px">'
        f'欢迎使用 {_h.escape(from_name)}。点击下方按钮完成邮箱确认，'
        '<b>链接 30 分钟内有效</b>；30 分钟内没有确认的账号会被自动清理，届时请重新注册。</p>'
        f'<div style="text-align:center;margin:24px 0">'
        f'<a href="{hl}" style="display:inline-block;background:#0b5fff;color:#fff;'
        'text-decoration:none;font-size:15px;padding:12px 28px;border-radius:8px">'
        '点击确认我的邮箱</a></div>'
        f'<p style="font-size:12px;color:#8a93a6;line-height:1.7;margin:18px 0 0">'
        f'如果按钮不可点，请复制下面的完整链接到浏览器打开：<br>'
        f'<a href="{hl}" style="color:#0b5fff;word-break:break-all">{hl}</a></p>'
        '<p style="font-size:12px;color:#b0b8c8;margin:14px 0 0">'
        '如果这不是你本人操作的注册，请直接忽略这封邮件。</p>'
        '</div></div>'
    )
    return _send(store, to_email, subject, body, html=html)


def send_test_email(store, to_email: str) -> dict:
    """发一封测试邮件，验证 SMTP 配置（控制台「邮箱发信」页的测试按钮）。"""
    d = mail_settings(store)
    subject = "LLM Relay 发信测试"
    body = ("这是一封 LLM Relay 发出来的测试邮件。\n\n"
            f"SMTP 服务器：{d['host']}:{d['port']}（STARTTLS={'开' if d['starttls'] else '关'}）\n"
            "收到说明发信配置正常，邮箱确认链接可以正常发送。\n")
    r = _send(store, to_email, subject, body)
    if r["ok"] and r["mode"] == "log":
        r["error"] = "当前为日志模式：没有实际发信，测试内容已写入服务器日志（搜 [mail/log]）。"
    return r


def send_password_reset_email(store, to_email: str, new_password: str,
                              base_url: str, from_name: str = "LLM Relay") -> dict:
    """发「密码已被管理员重置」邮件（v1.1.0：重置密码自动生成后通过邮件给账号本人）。

    日志模式下新密码只写进服务器日志（搜 [mail/log]）；smtp 模式发真邮件。
    返回值同 _send：{"mode","ok","error"}。"""
    subject = f"{from_name}：你的密码已被管理员重置"
    body = (f"你好，\n\n{from_name} 的中转网关管理员刚刚重置了你账号（{to_email}）的登录密码。\n\n"
            f"新密码：{new_password}\n"
            f"控制台：{base_url}\n\n"
            f"请尽快登录并在「设置 → 修改密码」改成只有你自己知道的密码。\n"
            f"如果不是你本人操作，请立即联系管理员。\n")
    r = _send(store, to_email, subject, body)
    if r["ok"] and r["mode"] == "log":
        r["error"] = "当前为日志模式：新密码已写入服务器日志（搜 [mail/log]）。"
    return r
