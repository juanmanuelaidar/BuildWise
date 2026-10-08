#!/usr/bin/env python3
import json
import os
import sys
from decimal import Decimal
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE_URL = os.getenv("BUILDWISE_API_URL", "http://127.0.0.1:8000").rstrip("/")


def request(method: str, path: str, *, token: str | None = None, payload: dict | None = None) -> tuple[int, object]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(payload).encode() if payload is not None else None
    try:
        with urlopen(Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method), timeout=60) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else None
    except HTTPError as exc:
        raw = exc.read()
        return exc.code, json.loads(raw) if raw else None


def expect(method: str, path: str, status: int, **kwargs) -> object:
    actual, payload = request(method, path, **kwargs)
    if actual != status:
        raise AssertionError(f"{method} {path}: esperado {status}, recibido {actual}: {payload}")
    return payload


def login(username: str, password: str) -> str:
    payload = expect("POST", "/auth/login", 200, payload={"username": username, "password": password})
    return payload["access_token"]


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Falta la variable obligatoria {name}")
    return value


def main() -> None:
    expect("GET", "/health", 200)
    admin_username = required_env("SMOKE_ADMIN_USERNAME")
    admin_token = login(admin_username, required_env("SMOKE_ADMIN_PASSWORD"))
    expect("POST", "/auth/login", 401, payload={"username": admin_username, "password": "invalid-smoke-password"})
    client_token = login(required_env("SMOKE_CLIENT_USERNAME"), required_env("SMOKE_CLIENT_PASSWORD"))
    expect("GET", "/auth/usuarios", 403, token=client_token)
    expect("GET", "/auth/usuarios", 200, token=admin_token)

    materials = expect("GET", "/materiales", 200)
    names = {material["nombre"] for material in materials}
    required = {"Cemento Portland", "Pastina", "Membrana Megaflex"}
    if not required.issubset(names):
        raise AssertionError(f"Faltan materiales MVP: {sorted(required - names)}")
    cement = next(material for material in materials if material["nombre"] == "Cemento Portland")
    history = expect("GET", f"/materiales/{cement['id']}/precios", 200, token=client_token)
    if not history:
        raise AssertionError("El historico de Cemento Portland esta vacio")

    mvp_materials = [material for material in materials if material["nombre"] in required]
    for material in mvp_materials:
        forecast = expect("GET", f"/materiales/{material['id']}/forecast?horizonte_meses=3", 200, token=client_token)
        if len(forecast.get("puntos", [])) != 3:
            raise AssertionError(f"Forecast incompleto para {material['nombre']}")
        selection = forecast.get("seleccion_modelo", {})
        if not selection.get("no_calibrado") or not selection.get("advertencia"):
            raise AssertionError(f"Faltan limites de calibracion para {material['nombre']}")

    optimization = expect(
        "POST", "/compras/optimizar-presupuesto", 200, token=client_token,
        payload={
            "presupuesto_total": "15.00", "horizonte_meses": 3,
            "materiales": [{"material_id": material["id"], "cantidad_objetivo": "1.0000", "criticidad": "alta"} for material in mvp_materials],
        },
    )
    if len(optimization["items"]) != len(mvp_materials):
        raise AssertionError("La optimizacion excluyo materiales del bootstrap")
    if Decimal(optimization["presupuesto_utilizado"]) > Decimal("15.00"):
        raise AssertionError("El presupuesto publicado excede el presupuesto disponible")
    for item in optimization["items"]:
        if Decimal(item["cantidad_recomendada_comprar_ahora"]) + Decimal(item["cantidad_recomendada_postergar"]) != Decimal(item["cantidad_objetivo"]):
            raise AssertionError("La optimizacion no conserva la cantidad objetivo")

    conversation = expect("POST", "/chat/conversaciones", 201, token=client_token, payload={"titulo": "Smoke API"})
    conversation_id = conversation["id"]
    expect("GET", f"/chat/conversaciones/{conversation_id}/mensajes", 200, token=client_token)
    expect(
        "PATCH",
        f"/chat/conversaciones/{conversation_id}",
        200,
        token=client_token,
        payload={"archived": True},
    )

    catalog_answer = expect("POST", "/chat/consultas", 200, token=client_token, payload={"pregunta": "que materiales hay?"})
    if catalog_answer.get("proveedor_utilizado"):
        raise AssertionError("La consulta deterministica de catalogo no debe depender de un LLM")

    deterministic_help = {
        "que puedo hacer con este asistente": "consultar materiales y precios",
        "que significa mape": "error porcentual promedio",
        "que es una anomalia": "variacion de precio fuera",
        "como funciona la recomendacion": "la calcula BuildWise, no la IA",
    }
    for question, expected_text in deterministic_help.items():
        answer = expect("POST", "/chat/consultas", 200, token=client_token, payload={"pregunta": question})
        if answer.get("proveedor_utilizado"):
            raise AssertionError(f"La ayuda deterministica invoco al proveedor: {question}")
        if expected_text not in answer.get("respuesta", ""):
            raise AssertionError(f"Respuesta de ayuda inesperada para: {question}")

    interpreted = expect(
        "POST",
        "/chat/presupuestacion/interpretar",
        200,
        token=client_token,
        payload={"necesidad": "Necesito 30 bolsas de cemento a 3 meses"},
    )
    if interpreted.get("cantidad") != "30" or interpreted.get("presupuesto_maximo") is not None:
        raise AssertionError(f"Interpretacion comercial parcial inesperada: {interpreted}")

    print("BuildWise API smoke: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"BuildWise API smoke: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
