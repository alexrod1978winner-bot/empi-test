from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from gigachat import GigaChat

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


# =========================
# CONFIG
# =========================

APP_TITLE = "EMPI MVP Backend"
DATA_DIR = Path("./data")
EXPORT_DIR = DATA_DIR / "exports"
SESSIONS_FILE = DATA_DIR / "sessions.json"

# ВАЖНО:
# Сюда вставь свой ключ авторизации GigaChat.
# Это должен быть именно credentials / authorization key,
# а не access token.
GIGACHAT_CREDENTIALS = "MDE5YzY2YWUtZmU5My03ODZiLTlkM2ItZDZlZDljYzJhNDRjOjBiNDU3N2NmLTU0Y2QtNGVhZC04MDgzLTA1MjEyYWY3N2QxYg=="

# Обычно для личного использования
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

# Можно начать с обычной модели
GIGACHAT_MODEL = "GigaChat"

DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
if not SESSIONS_FILE.exists():
    SESSIONS_FILE.write_text("{}", encoding="utf-8")


# =========================
# APP
# =========================

app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODELS
# =========================

class StartSessionRequest(BaseModel):
    tester_name: str = Field(min_length=1, max_length=100)
    scenario_type: str = Field(default="free_dialog", max_length=100)
    test_goal: str = Field(default="", max_length=1000)
    model_version: str = Field(default="10.6.4", max_length=100)
    llm_provider: str = Field(default="gigachat", max_length=100)
    params: str = Field(default="", max_length=10000)


class StartSessionResponse(BaseModel):
    session_id: str
    status: Literal["active"]
    model_version: str
    llm_provider: str
    welcome_message: str


class RunTurnRequest(BaseModel):
    session_id: str = Field(min_length=1)
    user_input: str = Field(min_length=1, max_length=12000)
    turn_index: int = Field(ge=1)


class RunTurnResponse(BaseModel):
    turn_index: int
    model_response: str
    latency_ms: int
    status_text: str = "Ответ получен"
    metrics: dict[str, Any] | None = None


class RewriteLastRequest(BaseModel):
    session_id: str = Field(min_length=1)
    based_on_turn: int = Field(ge=0)


class RewriteLastResponse(BaseModel):
    turn_index: int
    model_response: str
    latency_ms: int


class EndSessionRequest(BaseModel):
    session_id: str = Field(min_length=1)


class EndSessionResponse(BaseModel):
    session_id: str
    status: Literal["ended"]
    ended_at: str


class FeedbackPayload(BaseModel):
    alive_score: int | None = Field(default=None, ge=1, le=5)
    clarity_score: int | None = Field(default=None, ge=1, le=5)
    density_score: int | None = Field(default=None, ge=1, le=5)
    understanding_score: int | None = Field(default=None, ge=1, le=5)
    comment: str = Field(default="", max_length=4000)


class SubmitFeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1)
    feedback: FeedbackPayload


class SubmitFeedbackResponse(BaseModel):
    session_id: str
    status: Literal["feedback_saved"]


class MessageRecord(BaseModel):
    role: Literal["user", "assistant"]
    text: str
    turn_index: int
    timestamp: str
    latency_ms: int | None = None
    rewritten: bool = False


class SessionRecord(BaseModel):
    session_id: str
    tester_name: str
    scenario_type: str
    test_goal: str
    model_version: str
    llm_provider: str
    params: str
    profile_summary: str = ""
    status: Literal["active", "ended"]
    started_at: str
    ended_at: str | None = None
    messages: list[MessageRecord] = Field(default_factory=list)
    feedback: FeedbackPayload | None = None


# =========================
# STORAGE
# =========================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_sessions() -> dict[str, Any]:
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("sessions.json повреждён") from exc


def save_sessions(data: dict[str, Any]) -> None:
    SESSIONS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_session_or_404(session_id: str) -> dict[str, Any]:
    data = load_sessions()
    session = data.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail={"error": "session_not_found", "message": "Сессия не найдена"},
        )
    return session


def save_session_record(session: dict[str, Any]) -> None:
    data = load_sessions()
    data[session["session_id"]] = session
    save_sessions(data)


# =========================
# PARAMS / SURVEY
# =========================

def parse_params_json(params_value: str) -> dict[str, Any]:
    if not params_value:
        return {}
    try:
        data = json.loads(params_value)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_profile_summary_from_survey(params_value: str) -> str:
    params_data = parse_params_json(params_value)
    survey = params_data.get("survey", {})

    if not isinstance(survey, dict):
        return ""

    age_map = {
        "under_50": "менее 50 лет",
        "50_60": "50–60 лет",
        "60_70": "60–70 лет",
        "70_80": "70–80 лет",
        "80_plus": "более 80 лет",
    }

    yes_no_map = {
        "yes": "да",
        "no": "нет",
    }

    time_place_map = {
        "home": "чаще проводит время дома",
        "outside": "чаще проводит время вне дома",
        "no_matter": "место проведения времени не принципиально",
    }

    parts = []

    age_group = survey.get("age_group")
    if age_group in age_map:
        parts.append(f"Возрастная группа: {age_map[age_group]}.")

    retired = survey.get("retired")
    if retired in yes_no_map:
        parts.append(f"На пенсии: {yes_no_map[retired]}.")

    time_place = survey.get("time_place")
    if time_place in time_place_map:
        parts.append(f"Образ жизни: {time_place_map[time_place]}.")

    partner = survey.get("partner")
    if partner in yes_no_map:
        parts.append(f"Есть супруг(а): {yes_no_map[partner]}.")

    pet = survey.get("pet")
    if pet in yes_no_map:
        parts.append(f"Есть домашний питомец: {yes_no_map[pet]}.")

    return " ".join(parts)


def build_survey_prompt_context(session: dict[str, Any]) -> str:
    profile_summary = session.get("profile_summary", "").strip()
    if not profile_summary:
      return ""

    return (
        "Дополнительный мягкий контекст о пользователе из короткой анкеты. "
        "Используй его бережно, без навязчивости, без сухого перечисления фактов "
        "и без ощущения, что ты читаешь анкету вслух. "
        f"{profile_summary}"
    )


def build_welcome_message(session: dict[str, Any]) -> str:
    profile = session.get("profile_summary", "")

    if not profile:
        return "Здравствуйте. Я рядом. Можем просто поговорить."

    if "более 80 лет" in profile:
        return "Здравствуйте. Я рядом. Будем говорить спокойно и без спешки."
    if "чаще проводит время дома" in profile:
        return "Здравствуйте. Я рядом. Можем спокойно поговорить в удобном для вас ритме."
    if "Есть домашний питомец: да" in profile:
        return "Здравствуйте. Я рядом. Можем просто поговорить — спокойно и по-человечески."
    if "На пенсии: да" in profile:
        return "Здравствуйте. Я рядом. Можем поговорить в спокойном и удобном темпе."

    return "Здравствуйте. Я рядом. Можем просто поговорить."


# =========================
# GIGACHAT
# =========================

def build_system_prompt(session: dict[str, Any], rewrite: bool = False) -> str:
    survey_context = build_survey_prompt_context(session)

    base = (
        "Ты — ЭМПИ, адаптивный собеседник. "
        "Говори по-человечески, тепло, но без приторности. "
        "Не скатывайся в лекционность и не перегружай объяснениями. "
        "Не повторяй мысль пользователя без добавления нового смысла. "
        "Каждый ответ должен делать хотя бы одно из трёх: "
        "1) продвигать понимание, "
        "2) задавать полезный уточняющий вопрос, "
        "3) предлагать уместный следующий шаг. "
        "Старайся не размазывать мысль. "
        "Лучше плотнее и точнее, чем длиннее и красивее. "
        "Если пользователь спросит, учитываешь ли ты короткую анкету, "
        "отвечай честно: да, если её данные были переданы в контекст этого диалога. "
        "Не говори, что не видишь анкету, если её данные уже переданы тебе в системной инструкции. "
        "Не перечисляй анкетные пункты сухим списком, если пользователь прямо не просит. "
        f"Версия модели: {session['model_version']}. "
        f"Сценарий теста: {session['scenario_type']}. "
        f"Цель теста: {session['test_goal'] or 'не указана'}. "
    )

    if survey_context:
        base += survey_context + " "

    if rewrite:
        base += (
            "Ответь иначе на последний пользовательский ход. "
            "Не повторяй уже сказанное почти теми же словами. "
            "Дай более живой и полезный вариант."
        )

    return base


def build_gigachat_messages(
    session: dict[str, Any],
    user_input: str,
    rewrite: bool = False
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(session, rewrite=rewrite)}
    ]

    for msg in session["messages"]:
        if msg["role"] in {"user", "assistant"}:
            messages.append({"role": msg["role"], "content": msg["text"]})

    if not rewrite:
        messages.append({"role": "user", "content": user_input})

    return messages


def call_gigachat(messages: list[dict[str, str]]) -> str:
    if not GIGACHAT_CREDENTIALS or GIGACHAT_CREDENTIALS == "PASTE_YOUR_GIGACHAT_AUTH_KEY_HERE":
        raise RuntimeError("Не задан GIGACHAT_CREDENTIALS")

    with GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        scope=GIGACHAT_SCOPE,
        model=GIGACHAT_MODEL,
        verify_ssl_certs=False,
    ) as giga:
        response = giga.chat({"messages": messages})

    return response.choices[0].message.content


def run_model(session: dict[str, Any], user_input: str, rewrite: bool = False) -> tuple[str, int]:
    start = time.perf_counter()
    messages = build_gigachat_messages(session, user_input, rewrite=rewrite)
    output = call_gigachat(messages)
    latency_ms = int((time.perf_counter() - start) * 1000)
    return output, latency_ms


# =========================
# ROUTES
# =========================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "provider": "gigachat"}


@app.post("/start-session", response_model=StartSessionResponse)
def start_session(payload: StartSessionRequest) -> StartSessionResponse:
    session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    profile_summary = build_profile_summary_from_survey(payload.params)

    session = SessionRecord(
        session_id=session_id,
        tester_name=payload.tester_name,
        scenario_type=payload.scenario_type,
        test_goal=payload.test_goal,
        model_version=payload.model_version,
        llm_provider="gigachat",
        params=payload.params,
        profile_summary=profile_summary,
        status="active",
        started_at=now_iso(),
        messages=[],
    )

    session_dict = session.model_dump()
    save_session_record(session_dict)

    return StartSessionResponse(
        session_id=session_id,
        status="active",
        model_version=payload.model_version,
        llm_provider="gigachat",
        welcome_message=build_welcome_message(session_dict),
    )


@app.post("/run-turn", response_model=RunTurnResponse)
def run_turn(payload: RunTurnRequest) -> RunTurnResponse:
    session = get_session_or_404(payload.session_id)

    if session["status"] != "active":
        raise HTTPException(
            status_code=400,
            detail={"error": "session_already_ended", "message": "Сессия уже завершена"},
        )

    user_text = payload.user_input.strip()
    if not user_text:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_input", "message": "Сообщение пустое"},
        )

    session["messages"].append(
        MessageRecord(
            role="user",
            text=user_text,
            turn_index=payload.turn_index,
            timestamp=now_iso(),
        ).model_dump()
    )

    model_response, latency_ms = run_model(session, user_text)

    session["messages"].append(
        MessageRecord(
            role="assistant",
            text=model_response,
            turn_index=payload.turn_index,
            timestamp=now_iso(),
            latency_ms=latency_ms,
        ).model_dump()
    )

    save_session_record(session)

    return RunTurnResponse(
        turn_index=payload.turn_index,
        model_response=model_response,
        latency_ms=latency_ms,
        metrics={"score": 0.0},
    )


@app.post("/rewrite-last", response_model=RewriteLastResponse)
def rewrite_last(payload: RewriteLastRequest) -> RewriteLastResponse:
    session = get_session_or_404(payload.session_id)

    if session["status"] != "active":
        raise HTTPException(
            status_code=400,
            detail={"error": "session_already_ended", "message": "Сессия уже завершена"},
        )

    based_on_turn = payload.based_on_turn
    matching_user = None

    for msg in reversed(session["messages"]):
        if msg["role"] == "user" and msg["turn_index"] == based_on_turn:
            matching_user = msg
            break

    if not matching_user:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "turn_not_found",
                "message": "Не найден пользовательский ход для переформулировки"
            },
        )

    model_response, latency_ms = run_model(session, matching_user["text"], rewrite=True)

    session["messages"].append(
        MessageRecord(
            role="assistant",
            text=model_response,
            turn_index=based_on_turn,
            timestamp=now_iso(),
            latency_ms=latency_ms,
            rewritten=True,
        ).model_dump()
    )

    save_session_record(session)

    return RewriteLastResponse(
        turn_index=based_on_turn,
        model_response=model_response,
        latency_ms=latency_ms,
    )


@app.post("/end-session", response_model=EndSessionResponse)
def end_session(payload: EndSessionRequest) -> EndSessionResponse:
    session = get_session_or_404(payload.session_id)

    if session["status"] == "ended":
        return EndSessionResponse(
            session_id=payload.session_id,
            status="ended",
            ended_at=session["ended_at"] or now_iso(),
        )

    session["status"] = "ended"
    session["ended_at"] = now_iso()
    save_session_record(session)

    return EndSessionResponse(
        session_id=payload.session_id,
        status="ended",
        ended_at=session["ended_at"],
    )


@app.post("/submit-feedback", response_model=SubmitFeedbackResponse)
def submit_feedback(payload: SubmitFeedbackRequest) -> SubmitFeedbackResponse:
    session = get_session_or_404(payload.session_id)
    session["feedback"] = payload.feedback.model_dump()
    save_session_record(session)

    return SubmitFeedbackResponse(
        session_id=payload.session_id,
        status="feedback_saved",
    )


@app.get("/export/{session_id}")
def export_session(session_id: str) -> FileResponse:
    session = get_session_or_404(session_id)
    export_path = EXPORT_DIR / f"{session_id}.json"
    export_path.write_text(
        json.dumps(session, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return FileResponse(
        path=export_path,
        filename=f"{session_id}.json",
        media_type="application/json",
    )