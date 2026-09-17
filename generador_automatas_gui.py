#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENERADOR DE AUTÓMATAS FINITOS
Reconocedor de palabras con AFN y AFD

Funciones principales:
1. Recibe una expresión regular.
2. Inserta concatenación explícita.
3. Convierte la ER a notación postfija.
4. Construye un AFN usando el método de Thompson.
5. Convierte el AFN a AFD usando construcción por subconjuntos.
6. Muestra paso a paso las transformaciones.
7. Permite probar cadenas ingresadas por el usuario.
8. Puede exportar los autómatas a archivos .dot de Graphviz.

Operadores soportados:
    |   unión
    *   cero o más repeticiones
    +   una o más repeticiones
    ?   cero o una aparición
    ()  agrupación

La concatenación es implícita:
    ab      equivale a a·b
    a(b|c) equivale a a·(b|c)

Símbolo epsilon:
    ε

Para usar literalmente un operador, escápelo con \\:
    \\*   representa el carácter '*'
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

EPSILON = "ε"
CONCAT = "·"

Token = Tuple[str, str]  # (tipo, valor)


# ============================================================
# UTILIDADES DE EXPRESIONES REGULARES
# ============================================================

def tokenizar(regex: str) -> List[Token]:
    """Convierte la cadena de ER en tokens."""
    tokens: List[Token] = []
    i = 0

    while i < len(regex):
        c = regex[i]

        if c.isspace():
            i += 1
            continue

        if c == "\\":
            if i + 1 >= len(regex):
                raise ValueError("La expresión termina con '\\' sin un carácter para escapar.")
            tokens.append(("LIT", regex[i + 1]))
            i += 2
            continue

        if c == EPSILON:
            tokens.append(("EPS", EPSILON))
        elif c == "(":
            tokens.append(("LPAREN", c))
        elif c == ")":
            tokens.append(("RPAREN", c))
        elif c in {"|", "*", "+", "?"}:
            tokens.append(("OP", c))
        else:
            tokens.append(("LIT", c))

        i += 1

    if not tokens:
        raise ValueError("La expresión regular está vacía.")

    return tokens


def token_a_texto(token: Token) -> str:
    tipo, valor = token
    if tipo == "LIT":
        if valor in {"|", "*", "+", "?", "(", ")", "\\", CONCAT}:
            return "\\" + valor
        return valor
    return valor


def tokens_a_texto(tokens: Iterable[Token]) -> str:
    return "".join(token_a_texto(t) for t in tokens)


def puede_terminar(token: Token) -> bool:
    tipo, valor = token
    return (
        tipo in {"LIT", "EPS", "RPAREN"}
        or (tipo == "OP" and valor in {"*", "+", "?"})
    )


def puede_empezar(token: Token) -> bool:
    tipo, _ = token
    return tipo in {"LIT", "EPS", "LPAREN"}


def insertar_concatenacion(tokens: List[Token]) -> List[Token]:
    """Inserta el operador explícito · cuando la concatenación es implícita."""
    resultado: List[Token] = []

    for i, token in enumerate(tokens):
        if i > 0:
            anterior = tokens[i - 1]
            if puede_terminar(anterior) and puede_empezar(token):
                resultado.append(("OP", CONCAT))
        resultado.append(token)

    return resultado


def validar_sintaxis_basica(tokens: List[Token]) -> None:
    """Detecta errores comunes antes de convertir a postfija."""
    balance = 0
    esperando_operando = True
    anterior: Token | None = None

    for token in tokens:
        tipo, valor = token

        if tipo in {"LIT", "EPS"}:
            if not esperando_operando:
                raise ValueError(
                    "Falta un operador entre dos operandos. "
                    "Esto normalmente indica un error al interpretar la expresión."
                )
            esperando_operando = False

        elif tipo == "LPAREN":
            if not esperando_operando:
                raise ValueError("Falta un operador antes de '('.")
            balance += 1
            esperando_operando = True

        elif tipo == "RPAREN":
            if esperando_operando:
                raise ValueError("Paréntesis ')' vacío o colocado después de un operador.")
            balance -= 1
            if balance < 0:
                raise ValueError("Hay un ')' sin su '(' correspondiente.")
            esperando_operando = False

        elif tipo == "OP":
            if valor in {"*", "+", "?"}:
                if esperando_operando:
                    raise ValueError(f"El operador '{valor}' no tiene un operando a la izquierda.")
                # Sigue siendo posible cerrar, concatenar o unir.
                esperando_operando = False

            elif valor in {"|", CONCAT}:
                if esperando_operando:
                    raise ValueError(f"El operador '{valor}' no tiene un operando a la izquierda.")
                esperando_operando = True

        anterior = token

    if balance != 0:
        raise ValueError("Los paréntesis no están balanceados.")

    if esperando_operando:
        raise ValueError("La expresión termina con un operador incompleto.")


def infija_a_postfija(tokens: List[Token]) -> List[Token]:
    """Algoritmo tipo Shunting Yard para ER."""
    validar_sintaxis_basica(tokens)

    salida: List[Token] = []
    pila: List[Token] = []

    precedencia = {
        "|": 1,
        CONCAT: 2,
    }

    for token in tokens:
        tipo, valor = token

        if tipo in {"LIT", "EPS"}:
            salida.append(token)

        elif tipo == "LPAREN":
            pila.append(token)

        elif tipo == "RPAREN":
            while pila and pila[-1][0] != "LPAREN":
                salida.append(pila.pop())

            if not pila:
                raise ValueError("Paréntesis desbalanceados.")

            pila.pop()  # quitar '('

        elif tipo == "OP":
            if valor in {"*", "+", "?"}:
                # Operadores unarios postfijos: van directamente a la salida.
                salida.append(token)
            else:
                while (
                    pila
                    and pila[-1][0] == "OP"
                    and pila[-1][1] in precedencia
                    and precedencia[pila[-1][1]] >= precedencia[valor]
                ):
                    salida.append(pila.pop())

                pila.append(token)

    while pila:
        token = pila.pop()
        if token[0] in {"LPAREN", "RPAREN"}:
            raise ValueError("Paréntesis desbalanceados.")
        salida.append(token)

    return salida


# ============================================================
# AFN - MÉTODO DE THOMPSON
# ============================================================

@dataclass
class Fragmento:
    inicio: int
    fin: int


@dataclass
class AFN:
    estados: Set[int]
    alfabeto: Set[str]
    inicio: int
    aceptacion: int
    transiciones: Dict[int, Dict[str, Set[int]]]
    pasos: List[str]

    def destinos(self, estado: int, simbolo: str) -> Set[int]:
        return set(self.transiciones.get(estado, {}).get(simbolo, set()))


class ConstructorThompson:
    def __init__(self) -> None:
        self.contador = 0
        self.transiciones: Dict[int, Dict[str, Set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.estados: Set[int] = set()
        self.alfabeto: Set[str] = set()
        self.pasos: List[str] = []

    def nuevo_estado(self) -> int:
        estado = self.contador
        self.contador += 1
        self.estados.add(estado)
        return estado

    def agregar_transicion(self, origen: int, simbolo: str, destino: int) -> None:
        self.transiciones[origen][simbolo].add(destino)
        if simbolo != EPSILON:
            self.alfabeto.add(simbolo)

    def construir(self, postfija: List[Token]) -> AFN:
        pila: List[Fragmento] = []

        for numero_paso, token in enumerate(postfija, start=1):
            tipo, valor = token

            if tipo == "LIT":
                inicio = self.nuevo_estado()
                fin = self.nuevo_estado()
                self.agregar_transicion(inicio, valor, fin)
                pila.append(Fragmento(inicio, fin))

                self.pasos.append(
                    f"{numero_paso}. Símbolo '{valor}': "
                    f"crear q{inicio} --{valor}--> q{fin}"
                )

            elif tipo == "EPS":
                inicio = self.nuevo_estado()
                fin = self.nuevo_estado()
                self.agregar_transicion(inicio, EPSILON, fin)
                pila.append(Fragmento(inicio, fin))

                self.pasos.append(
                    f"{numero_paso}. Epsilon: "
                    f"crear q{inicio} --{EPSILON}--> q{fin}"
                )

            elif tipo == "OP":
                if valor == CONCAT:
                    if len(pila) < 2:
                        raise ValueError("Concatenación inválida en la expresión.")

                    derecha = pila.pop()
                    izquierda = pila.pop()

                    self.agregar_transicion(
                        izquierda.fin, EPSILON, derecha.inicio
                    )
                    nuevo = Fragmento(izquierda.inicio, derecha.fin)
                    pila.append(nuevo)

                    self.pasos.append(
                        f"{numero_paso}. Concatenación: conectar "
                        f"q{izquierda.fin} --{EPSILON}--> q{derecha.inicio}. "
                        f"Fragmento resultante: q{nuevo.inicio} ... q{nuevo.fin}"
                    )

                elif valor == "|":
                    if len(pila) < 2:
                        raise ValueError("Unión inválida en la expresión.")

                    derecha = pila.pop()
                    izquierda = pila.pop()

                    inicio = self.nuevo_estado()
                    fin = self.nuevo_estado()

                    self.agregar_transicion(inicio, EPSILON, izquierda.inicio)
                    self.agregar_transicion(inicio, EPSILON, derecha.inicio)
                    self.agregar_transicion(izquierda.fin, EPSILON, fin)
                    self.agregar_transicion(derecha.fin, EPSILON, fin)

                    pila.append(Fragmento(inicio, fin))

                    self.pasos.append(
                        f"{numero_paso}. Unión '|': crear q{inicio} y q{fin}; "
                        f"q{inicio} se conecta por {EPSILON} con "
                        f"q{izquierda.inicio} y q{derecha.inicio}; "
                        f"q{izquierda.fin} y q{derecha.fin} se conectan "
                        f"por {EPSILON} con q{fin}."
                    )

                elif valor == "*":
                    if len(pila) < 1:
                        raise ValueError("Cerradura '*' inválida.")

                    frag = pila.pop()

                    inicio = self.nuevo_estado()
                    fin = self.nuevo_estado()

                    self.agregar_transicion(inicio, EPSILON, frag.inicio)
                    self.agregar_transicion(inicio, EPSILON, fin)
                    self.agregar_transicion(frag.fin, EPSILON, frag.inicio)
                    self.agregar_transicion(frag.fin, EPSILON, fin)

                    pila.append(Fragmento(inicio, fin))

                    self.pasos.append(
                        f"{numero_paso}. Cerradura '*': crear q{inicio} y q{fin}; "
                        f"permitir entrar, repetir o saltar el fragmento."
                    )

                elif valor == "+":
                    if len(pila) < 1:
                        raise ValueError("Cerradura '+' inválida.")

                    frag = pila.pop()

                    inicio = self.nuevo_estado()
                    fin = self.nuevo_estado()

                    self.agregar_transicion(inicio, EPSILON, frag.inicio)
                    self.agregar_transicion(frag.fin, EPSILON, frag.inicio)
                    self.agregar_transicion(frag.fin, EPSILON, fin)

                    pila.append(Fragmento(inicio, fin))

                    self.pasos.append(
                        f"{numero_paso}. Cerradura '+': crear q{inicio} y q{fin}; "
                        f"el fragmento debe ejecutarse al menos una vez."
                    )

                elif valor == "?":
                    if len(pila) < 1:
                        raise ValueError("Operador '?' inválido.")

                    frag = pila.pop()

                    inicio = self.nuevo_estado()
                    fin = self.nuevo_estado()

                    self.agregar_transicion(inicio, EPSILON, frag.inicio)
                    self.agregar_transicion(inicio, EPSILON, fin)
                    self.agregar_transicion(frag.fin, EPSILON, fin)

                    pila.append(Fragmento(inicio, fin))

                    self.pasos.append(
                        f"{numero_paso}. Operador '?': crear q{inicio} y q{fin}; "
                        f"el fragmento puede aparecer cero o una vez."
                    )

        if len(pila) != 1:
            raise ValueError(
                "La expresión regular no pudo reducirse a un único autómata."
            )

        resultado = pila.pop()

        # Convertimos defaultdict anidados a diccionarios normales.
        trans_norm: Dict[int, Dict[str, Set[int]]] = {}
        for origen, por_simbolo in self.transiciones.items():
            trans_norm[origen] = {
                simbolo: set(destinos)
                for simbolo, destinos in por_simbolo.items()
            }

        return AFN(
            estados=set(self.estados),
            alfabeto=set(self.alfabeto),
            inicio=resultado.inicio,
            aceptacion=resultado.fin,
            transiciones=trans_norm,
            pasos=list(self.pasos),
        )


# ============================================================
# AFD - CONSTRUCCIÓN POR SUBCONJUNTOS
# ============================================================

@dataclass
class AFD:
    alfabeto: List[str]
    estados: List[FrozenSet[int]]
    nombres: Dict[FrozenSet[int], str]
    inicio: FrozenSet[int]
    aceptacion: Set[FrozenSet[int]]
    transiciones: Dict[FrozenSet[int], Dict[str, FrozenSet[int]]]
    pasos: List[str]


def epsilon_cerradura(afn: AFN, estados: Iterable[int]) -> FrozenSet[int]:
    cierre = set(estados)
    pila = list(estados)

    while pila:
        estado = pila.pop()
        for destino in afn.destinos(estado, EPSILON):
            if destino not in cierre:
                cierre.add(destino)
                pila.append(destino)

    return frozenset(cierre)


def mover(afn: AFN, estados: Iterable[int], simbolo: str) -> Set[int]:
    resultado: Set[int] = set()

    for estado in estados:
        resultado.update(afn.destinos(estado, simbolo))

    return resultado


def construir_afd(afn: AFN) -> AFD:
    alfabeto = sorted(afn.alfabeto)

    inicio = epsilon_cerradura(afn, {afn.inicio})

    cola: deque[FrozenSet[int]] = deque([inicio])
    estados: List[FrozenSet[int]] = [inicio]
    nombres: Dict[FrozenSet[int], str] = {inicio: "D0"}
    transiciones: Dict[FrozenSet[int], Dict[str, FrozenSet[int]]] = {}
    aceptacion: Set[FrozenSet[int]] = set()
    pasos: List[str] = []

    if afn.aceptacion in inicio:
        aceptacion.add(inicio)

    pasos.append(
        f"1. Estado inicial del AFD: D0 = "
        f"ε-cerradura({{q{afn.inicio}}}) = {formatear_conjunto(inicio)}"
    )

    numero_paso = 2

    while cola:
        actual = cola.popleft()
        transiciones[actual] = {}

        for simbolo in alfabeto:
            mov = mover(afn, actual, simbolo)
            destino = epsilon_cerradura(afn, mov)

            if destino not in nombres:
                nombre = f"D{len(estados)}"
                nombres[destino] = nombre
                estados.append(destino)
                cola.append(destino)

                if afn.aceptacion in destino:
                    aceptacion.add(destino)

                pasos.append(
                    f"{numero_paso}. Nuevo estado {nombre} = "
                    f"{formatear_conjunto(destino)}"
                )
                numero_paso += 1

            transiciones[actual][simbolo] = destino

            pasos.append(
                f"{numero_paso}. {nombres[actual]} con '{simbolo}': "
                f"mover = {formatear_conjunto(mov)}; "
                f"ε-cerradura = {nombres[destino]} "
                f"{formatear_conjunto(destino)}"
            )
            numero_paso += 1

    return AFD(
        alfabeto=alfabeto,
        estados=estados,
        nombres=nombres,
        inicio=inicio,
        aceptacion=aceptacion,
        transiciones=transiciones,
        pasos=pasos,
    )


# ============================================================
# IMPRESIÓN DE RESULTADOS
# ============================================================

def formatear_conjunto(estados: Iterable[int]) -> str:
    ordenados = sorted(estados)
    if not ordenados:
        return "∅"
    return "{" + ", ".join(f"q{x}" for x in ordenados) + "}"


def imprimir_separador(titulo: str) -> None:
    print("\n" + "=" * 72)
    print(titulo.center(72))
    print("=" * 72)


def imprimir_afn(afn: AFN) -> None:
    imprimir_separador("AFN RESULTANTE - MÉTODO DE THOMPSON")

    print(f"Estados: {formatear_conjunto(afn.estados)}")
    print(f"Alfabeto: {{{', '.join(sorted(afn.alfabeto))}}}")
    print(f"Estado inicial: q{afn.inicio}")
    print(f"Estado de aceptación: q{afn.aceptacion}")

    print("\nTransiciones:")
    hay_transiciones = False

    for origen in sorted(afn.estados):
        por_simbolo = afn.transiciones.get(origen, {})

        simbolos = sorted(
            por_simbolo.keys(),
            key=lambda x: (x != EPSILON, x)
        )

        for simbolo in simbolos:
            for destino in sorted(por_simbolo[simbolo]):
                print(f"  q{origen} --{simbolo}--> q{destino}")
                hay_transiciones = True

    if not hay_transiciones:
        print("  (sin transiciones)")


def imprimir_afd(afd: AFD) -> None:
    imprimir_separador("AFD RESULTANTE - CONSTRUCCIÓN POR SUBCONJUNTOS")

    print("Equivalencia entre estados del AFD y subconjuntos del AFN:")
    for estado in afd.estados:
        marca = "  [ACEPTACIÓN]" if estado in afd.aceptacion else ""
        if estado == afd.inicio:
            marca += "  [INICIAL]"
        print(
            f"  {afd.nombres[estado]} = "
            f"{formatear_conjunto(estado)}{marca}"
        )

    if not afd.alfabeto:
        print("\nEl alfabeto está vacío.")
        return

    print("\nTabla de transición del AFD:")

    encabezados = ["Estado"] + afd.alfabeto
    filas: List[List[str]] = []

    for estado in afd.estados:
        nombre = afd.nombres[estado]

        prefijo = ""
        if estado == afd.inicio:
            prefijo += "->"
        if estado in afd.aceptacion:
            prefijo += "*"

        fila = [prefijo + nombre]

        for simbolo in afd.alfabeto:
            destino = afd.transiciones[estado][simbolo]
            fila.append(afd.nombres[destino])

        filas.append(fila)

    anchos = [
        max(len(encabezados[i]), *(len(fila[i]) for fila in filas))
        for i in range(len(encabezados))
    ]

    def imprimir_fila(fila: List[str]) -> None:
        print(
            " | ".join(
                texto.ljust(anchos[i])
                for i, texto in enumerate(fila)
            )
        )

    imprimir_fila(encabezados)
    print("-+-".join("-" * ancho for ancho in anchos))

    for fila in filas:
        imprimir_fila(fila)

    print("\nLeyenda: -> inicial, * aceptación")


# ============================================================
# RECONOCIMIENTO DE CADENAS
# ============================================================

def probar_cadena(afd: AFD, cadena: str, mostrar_recorrido: bool = True) -> bool:
    actual = afd.inicio

    if mostrar_recorrido:
        print("\nRecorrido del AFD:")
        print(f"  Inicio en {afd.nombres[actual]}")

    if cadena == EPSILON:
        cadena = ""

    for posicion, simbolo in enumerate(cadena, start=1):
        if simbolo not in afd.alfabeto:
            if mostrar_recorrido:
                print(
                    f"  Símbolo #{posicion} '{simbolo}': "
                    "no pertenece al alfabeto."
                )
            return False

        destino = afd.transiciones[actual][simbolo]

        if mostrar_recorrido:
            print(
                f"  {afd.nombres[actual]} --{simbolo}--> "
                f"{afd.nombres[destino]}"
            )

        actual = destino

    aceptada = actual in afd.aceptacion

    if mostrar_recorrido:
        if aceptada:
            print(
                f"  Estado final: {afd.nombres[actual]} "
                "[ACEPTACIÓN]"
            )
        else:
            print(
                f"  Estado final: {afd.nombres[actual]} "
                "[NO ACEPTACIÓN]"
            )

    return aceptada


# ============================================================
# EXPORTACIÓN GRAPHVIZ (.DOT)
# ============================================================

def escapar_dot(texto: str) -> str:
    return texto.replace("\\", "\\\\").replace('"', '\\"')


def exportar_afn_dot(afn: AFN, archivo: str = "afn.dot") -> None:
    lineas = [
        "digraph AFN {",
        "  rankdir=LR;",
        '  node [shape=circle];',
        '  inicio [shape=point];',
        f'  inicio -> q{afn.inicio};',
        f'  q{afn.aceptacion} [shape=doublecircle];',
    ]

    for origen in sorted(afn.estados):
        por_simbolo = afn.transiciones.get(origen, {})
        for simbolo, destinos in por_simbolo.items():
            for destino in sorted(destinos):
                lineas.append(
                    f'  q{origen} -> q{destino} '
                    f'[label="{escapar_dot(simbolo)}"];'
                )

    lineas.append("}")

    with open(archivo, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))


def exportar_afd_dot(afd: AFD, archivo: str = "afd.dot") -> None:
    lineas = [
        "digraph AFD {",
        "  rankdir=LR;",
        '  node [shape=circle];',
        '  inicio [shape=point];',
        f'  inicio -> {afd.nombres[afd.inicio]};',
    ]

    for estado in afd.aceptacion:
        lineas.append(
            f'  {afd.nombres[estado]} [shape=doublecircle];'
        )

    for origen in afd.estados:
        for simbolo, destino in afd.transiciones[origen].items():
            lineas.append(
                f'  {afd.nombres[origen]} -> {afd.nombres[destino]} '
                f'[label="{escapar_dot(simbolo)}"];'
            )

    lineas.append("}")

    with open(archivo, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))


# ============================================================
# COMPILACIÓN COMPLETA DE ER
# ============================================================

@dataclass
class ResultadoCompilacion:
    regex_original: str
    tokens: List[Token]
    tokens_concat: List[Token]
    postfija: List[Token]
    afn: AFN
    afd: AFD


def compilar_regex(regex: str) -> ResultadoCompilacion:
    tokens = tokenizar(regex)
    tokens_concat = insertar_concatenacion(tokens)
    postfija = infija_a_postfija(tokens_concat)

    constructor = ConstructorThompson()
    afn = constructor.construir(postfija)

    afd = construir_afd(afn)

    return ResultadoCompilacion(
        regex_original=regex,
        tokens=tokens,
        tokens_concat=tokens_concat,
        postfija=postfija,
        afn=afn,
        afd=afd,
    )


def mostrar_transformacion(resultado: ResultadoCompilacion) -> None:
    imprimir_separador("TRANSFORMACIÓN DE LA EXPRESIÓN REGULAR")

    print(f"1. ER original:              {resultado.regex_original}")
    print(
        f"2. Concatenación explícita:  "
        f"{tokens_a_texto(resultado.tokens_concat)}"
    )
    print(
        f"3. Notación postfija:         "
        f"{tokens_a_texto(resultado.postfija)}"
    )

    imprimir_separador("PASOS DEL MÉTODO DE THOMPSON")
    for paso in resultado.afn.pasos:
        print(paso)

    imprimir_afn(resultado.afn)

    imprimir_separador("PASOS DE CONSTRUCCIÓN POR SUBCONJUNTOS")
    for paso in resultado.afd.pasos:
        print(paso)

    imprimir_afd(resultado.afd)


# ============================================================
# INTERFAZ DE CONSOLA
# ============================================================

def mostrar_ayuda() -> None:
    print(
        """
OPERADORES ADMITIDOS
--------------------
a        símbolo literal
ab       concatenación
a|b      unión
a*       cero o más repeticiones
a+       una o más repeticiones
a?       cero o una aparición
(a|b)    agrupación
ε        cadena vacía

EJEMPLOS
--------
(a|b)*abb
a(b|c)*
a+b?
(a|ε)b

Durante la prueba de cadenas:
- Presione ENTER para probar la cadena vacía.
- Escriba 'nueva' para ingresar otra expresión regular.
- Escriba 'dot' para exportar AFN y AFD a Graphviz.
- Escriba 'salir' para terminar.
"""
    )


def main() -> None:
    imprimir_separador("GENERADOR DE AUTÓMATAS FINITOS")
    print("ER -> AFN (Thompson) -> AFD (Subconjuntos)")
    mostrar_ayuda()

    while True:
        regex = input("\nIngrese una expresión regular: ").strip()

        if regex.lower() == "salir":
            print("\nPrograma finalizado.")
            break

        try:
            resultado = compilar_regex(regex)
        except ValueError as error:
            print(f"\n[ERROR] {error}")
            continue

        mostrar_transformacion(resultado)

        while True:
            print("\n" + "-" * 72)
            cadena = input(
                "Cadena a probar "
                "(ENTER=ε, nueva, dot, salir): "
            )

            comando = cadena.lower().strip()

            if comando == "salir":
                print("\nPrograma finalizado.")
                return

            if comando == "nueva":
                break

            if comando == "dot":
                exportar_afn_dot(resultado.afn)
                exportar_afd_dot(resultado.afd)
                print(
                    "\nSe generaron 'afn.dot' y 'afd.dot'. "
                    "Puede abrirlos con Graphviz."
                )
                continue

            aceptada = probar_cadena(
                resultado.afd,
                cadena,
                mostrar_recorrido=True
            )

            if aceptada:
                print("\nRESULTADO: CADENA ACEPTADA ✅")
            else:
                print("\nRESULTADO: CADENA RECHAZADA ❌")



# ============================================================
# INTERFAZ GRÁFICA (TKINTER)
# ============================================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math


class AplicacionAutomatas(tk.Tk):
    BG = "#111827"
    PANEL = "#1f2937"
    PANEL_2 = "#0f172a"
    TEXT = "#f9fafb"
    MUTED = "#9ca3af"
    ACCENT = "#6366f1"
    SUCCESS = "#22c55e"
    DANGER = "#ef4444"
    BORDER = "#374151"

    def __init__(self):
        super().__init__()
        self.title("Generador de Autómatas Finitos")
        self.geometry("1180x760")
        self.minsize(980, 650)
        self.configure(bg=self.BG)

        self.resultado = None

        self._configurar_estilos()
        self._crear_encabezado()
        self._crear_contenido()

    def _configurar_estilos(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TNotebook",
            background=self.BG,
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background=self.PANEL,
            foreground=self.TEXT,
            padding=(16, 9),
            font=("Arial", 11, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.ACCENT)],
            foreground=[("selected", "#ffffff")],
        )

        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="#ffffff",
            padding=(14, 8),
            font=("Arial", 11, "bold"),
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#4f46e5")]
        )

        style.configure(
            "Secondary.TButton",
            background=self.PANEL,
            foreground=self.TEXT,
            padding=(12, 8),
            font=("Arial", 10, "bold"),
            borderwidth=1,
        )

    def _crear_encabezado(self):
        header = tk.Frame(self, bg=self.BG, padx=24, pady=18)
        header.pack(fill="x")

        title_box = tk.Frame(header, bg=self.BG)
        title_box.pack(side="left")

        tk.Label(
            title_box,
            text="GENERADOR DE AUTÓMATAS FINITOS",
            bg=self.BG,
            fg=self.TEXT,
            font=("Arial", 22, "bold"),
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="Expresión Regular  →  AFN (Thompson)  →  AFD (Subconjuntos)",
            bg=self.BG,
            fg=self.MUTED,
            font=("Arial", 11),
        ).pack(anchor="w", pady=(4, 0))

        badge = tk.Label(
            header,
            text="PROYECTO DE AUTÓMATAS",
            bg=self.ACCENT,
            fg="#ffffff",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=7,
        )
        badge.pack(side="right")

    def _crear_contenido(self):
        main = tk.Frame(self, bg=self.BG, padx=24, pady=0)
        main.pack(fill="both", expand=True, pady=(0, 20))

        # Barra de entrada
        input_card = tk.Frame(
            main,
            bg=self.PANEL,
            highlightbackground=self.BORDER,
            highlightthickness=1,
            padx=18,
            pady=16,
        )
        input_card.pack(fill="x", pady=(0, 16))

        tk.Label(
            input_card,
            text="Expresión regular",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Arial", 11, "bold"),
        ).pack(anchor="w")

        row = tk.Frame(input_card, bg=self.PANEL)
        row.pack(fill="x", pady=(8, 0))

        self.regex_var = tk.StringVar(value="(a|b)*abb")
        self.regex_entry = tk.Entry(
            row,
            textvariable=self.regex_var,
            bg=self.PANEL_2,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            relief="flat",
            font=("Menlo", 13),
            highlightbackground=self.BORDER,
            highlightcolor=self.ACCENT,
            highlightthickness=1,
        )
        self.regex_entry.pack(side="left", fill="x", expand=True, ipady=9)
        self.regex_entry.bind("<Return>", lambda _e: self.generar())

        ttk.Button(
            row,
            text="Generar autómatas",
            style="Accent.TButton",
            command=self.generar,
        ).pack(side="left", padx=(12, 0))

        ttk.Button(
            row,
            text="Limpiar",
            style="Secondary.TButton",
            command=self.limpiar,
        ).pack(side="left", padx=(8, 0))

        tk.Label(
            input_card,
            text="Operadores:  |  unión    *  cerradura    +  una o más    ?  opcional    ()  agrupación    ε  vacío",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Arial", 9),
        ).pack(anchor="w", pady=(8, 0))

        # Pestañas
        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True)

        self.tab_resumen = self._nueva_tab("Resumen")
        self.tab_thompson = self._nueva_tab("Thompson")
        self.tab_afn = self._nueva_tab("AFN")
        self.tab_subconjuntos = self._nueva_tab("Subconjuntos")
        self.tab_afd = self._nueva_tab("AFD")
        self.tab_prueba = self._nueva_tab("Probar cadena")

        self._crear_tab_resumen()
        self._crear_tab_texto(self.tab_thompson, "texto_thompson")
        self._crear_tab_automata(self.tab_afn, "canvas_afn", "texto_afn")
        self._crear_tab_texto(self.tab_subconjuntos, "texto_subconjuntos")
        self._crear_tab_automata(self.tab_afd, "canvas_afd", "texto_afd")
        self._crear_tab_prueba()

        self._poner_bienvenida()

    def _nueva_tab(self, titulo):
        frame = tk.Frame(self.notebook, bg=self.BG)
        self.notebook.add(frame, text=titulo)
        return frame

    def _crear_tab_resumen(self):
        card = tk.Frame(self.tab_resumen, bg=self.PANEL, padx=20, pady=20)
        card.pack(fill="both", expand=True, padx=4, pady=4)

        self.resumen_texto = tk.Text(
            card,
            bg=self.PANEL_2,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            relief="flat",
            wrap="word",
            font=("Menlo", 11),
            padx=16,
            pady=16,
        )
        self.resumen_texto.pack(fill="both", expand=True)
        self.resumen_texto.config(state="disabled")

    def _crear_tab_texto(self, tab, atributo):
        cont = tk.Frame(tab, bg=self.PANEL, padx=14, pady=14)
        cont.pack(fill="both", expand=True, padx=4, pady=4)

        text = tk.Text(
            cont,
            bg=self.PANEL_2,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            relief="flat",
            wrap="word",
            font=("Menlo", 10),
            padx=14,
            pady=14,
        )
        scroll = tk.Scrollbar(cont, command=text.yview)
        text.configure(yscrollcommand=scroll.set)

        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)

        setattr(self, atributo, text)

    def _crear_tab_automata(self, tab, canvas_attr, texto_attr):
        paned = tk.PanedWindow(
            tab,
            orient="vertical",
            bg=self.BG,
            sashwidth=6,
            sashrelief="flat",
        )
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        top = tk.Frame(paned, bg=self.PANEL)
        bottom = tk.Frame(paned, bg=self.PANEL)
        paned.add(top, minsize=300)
        paned.add(bottom, minsize=180)

        canvas = tk.Canvas(
            top,
            bg="#f8fafc",
            highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True, padx=10, pady=10)
        setattr(self, canvas_attr, canvas)

        text = tk.Text(
            bottom,
            bg=self.PANEL_2,
            fg=self.TEXT,
            relief="flat",
            wrap="none",
            font=("Menlo", 10),
            padx=12,
            pady=12,
            height=9,
        )
        text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        setattr(self, texto_attr, text)

    def _crear_tab_prueba(self):
        outer = tk.Frame(self.tab_prueba, bg=self.BG, padx=4, pady=4)
        outer.pack(fill="both", expand=True)

        card = tk.Frame(
            outer,
            bg=self.PANEL,
            padx=24,
            pady=24,
        )
        card.pack(fill="both", expand=True)

        tk.Label(
            card,
            text="Probar una cadena en el AFD",
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Arial", 18, "bold"),
        ).pack(anchor="w")

        tk.Label(
            card,
            text="Deje el campo vacío para probar ε.",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Arial", 10),
        ).pack(anchor="w", pady=(4, 14))

        input_row = tk.Frame(card, bg=self.PANEL)
        input_row.pack(fill="x")

        self.cadena_var = tk.StringVar()
        self.cadena_entry = tk.Entry(
            input_row,
            textvariable=self.cadena_var,
            bg=self.PANEL_2,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            relief="flat",
            font=("Menlo", 13),
            highlightbackground=self.BORDER,
            highlightcolor=self.ACCENT,
            highlightthickness=1,
        )
        self.cadena_entry.pack(side="left", fill="x", expand=True, ipady=9)
        self.cadena_entry.bind("<Return>", lambda _e: self.probar())

        ttk.Button(
            input_row,
            text="Probar",
            style="Accent.TButton",
            command=self.probar,
        ).pack(side="left", padx=(10, 0))

        self.estado_prueba = tk.Label(
            card,
            text="Primero genere un autómata.",
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Arial", 14, "bold"),
        )
        self.estado_prueba.pack(anchor="w", pady=(22, 8))

        self.recorrido_texto = tk.Text(
            card,
            bg=self.PANEL_2,
            fg=self.TEXT,
            relief="flat",
            wrap="word",
            font=("Menlo", 11),
            padx=14,
            pady=14,
        )
        self.recorrido_texto.pack(fill="both", expand=True)

    def _poner_texto(self, widget, contenido):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", contenido)
        widget.config(state="disabled")

    def _poner_bienvenida(self):
        texto = (
            "BIENVENIDO\n\n"
            "1. Escriba una expresión regular en la parte superior.\n"
            "2. Presione “Generar autómatas”.\n"
            "3. Revise cada pestaña para observar el proceso completo.\n"
            "4. En “Probar cadena” puede comprobar si una entrada pertenece al lenguaje.\n\n"
            "Ejemplo recomendado:\n"
            "    (a|b)*abb\n"
        )
        self._poner_texto(self.resumen_texto, texto)

    def generar(self):
        regex = self.regex_var.get().strip()

        if not regex:
            messagebox.showwarning(
                "Expresión vacía",
                "Ingrese una expresión regular."
            )
            return

        try:
            self.resultado = compilar_regex(regex)
        except Exception as error:
            messagebox.showerror(
                "Error en la expresión regular",
                str(error)
            )
            return

        r = self.resultado

        resumen = [
            "TRANSFORMACIÓN COMPLETA",
            "",
            f"ER original:             {r.regex_original}",
            f"Concatenación explícita: {tokens_a_texto(r.tokens_concat)}",
            f"Notación postfija:        {tokens_a_texto(r.postfija)}",
            "",
            f"Estados del AFN:          {len(r.afn.estados)}",
            f"Estado inicial AFN:       q{r.afn.inicio}",
            f"Estado final AFN:         q{r.afn.aceptacion}",
            "",
            f"Estados del AFD:          {len(r.afd.estados)}",
            f"Alfabeto:                 {', '.join(r.afd.alfabeto) if r.afd.alfabeto else '∅'}",
            "",
            "Use las demás pestañas para revisar el procedimiento paso a paso."
        ]
        self._poner_texto(self.resumen_texto, "\n".join(resumen))

        self._poner_texto(
            self.texto_thompson,
            "\n\n".join(r.afn.pasos)
        )

        self._poner_texto(
            self.texto_subconjuntos,
            "\n\n".join(r.afd.pasos)
        )

        self._poner_texto(
            self.texto_afn,
            self._texto_afn(r.afn)
        )

        self._poner_texto(
            self.texto_afd,
            self._texto_afd(r.afd)
        )

        self._dibujar_afn(r.afn)
        self._dibujar_afd(r.afd)

        self.estado_prueba.config(
            text="AFD listo. Ingrese una cadena.",
            fg=self.SUCCESS
        )
        self.recorrido_texto.delete("1.0", "end")
        self.recorrido_texto.insert(
            "1.0",
            "Autómata generado correctamente."
        )

        self.notebook.select(self.tab_resumen)

    def limpiar(self):
        self.resultado = None
        self.regex_var.set("")
        self.cadena_var.set("")

        for widget in [
            self.resumen_texto,
            self.texto_thompson,
            self.texto_afn,
            self.texto_subconjuntos,
            self.texto_afd,
        ]:
            self._poner_texto(widget, "")

        self.canvas_afn.delete("all")
        self.canvas_afd.delete("all")
        self.recorrido_texto.delete("1.0", "end")

        self.estado_prueba.config(
            text="Primero genere un autómata.",
            fg=self.MUTED
        )
        self.regex_entry.focus_set()

    def probar(self):
        if self.resultado is None:
            messagebox.showwarning(
                "Sin autómata",
                "Primero genere un AFD."
            )
            return

        cadena = self.cadena_var.get()
        afd = self.resultado.afd
        actual = afd.inicio

        recorrido = [
            f"Cadena: {cadena if cadena else 'ε'}",
            "",
            f"Inicio en {afd.nombres[actual]}"
        ]

        aceptada = True

        for i, simbolo in enumerate(cadena, start=1):
            if simbolo not in afd.alfabeto:
                recorrido.append(
                    f"Símbolo #{i}: '{simbolo}' no pertenece al alfabeto."
                )
                aceptada = False
                break

            destino = afd.transiciones[actual][simbolo]
            recorrido.append(
                f"{afd.nombres[actual]} --{simbolo}--> {afd.nombres[destino]}"
            )
            actual = destino

        if aceptada:
            aceptada = actual in afd.aceptacion

        recorrido.append("")
        if aceptada:
            recorrido.append("RESULTADO: CADENA ACEPTADA")
            self.estado_prueba.config(
                text="✓ CADENA ACEPTADA",
                fg=self.SUCCESS
            )
        else:
            recorrido.append("RESULTADO: CADENA RECHAZADA")
            self.estado_prueba.config(
                text="✗ CADENA RECHAZADA",
                fg=self.DANGER
            )

        self.recorrido_texto.delete("1.0", "end")
        self.recorrido_texto.insert("1.0", "\n".join(recorrido))

    def _texto_afn(self, afn):
        lineas = [
            f"Estados: {formatear_conjunto(afn.estados)}",
            f"Inicial: q{afn.inicio}",
            f"Aceptación: q{afn.aceptacion}",
            "",
            "TRANSICIONES:"
        ]

        for origen in sorted(afn.estados):
            por_simbolo = afn.transiciones.get(origen, {})
            for simbolo in sorted(
                por_simbolo.keys(),
                key=lambda x: (x != EPSILON, x)
            ):
                for destino in sorted(por_simbolo[simbolo]):
                    lineas.append(
                        f"q{origen} --{simbolo}--> q{destino}"
                    )
        return "\n".join(lineas)

    def _texto_afd(self, afd):
        lineas = ["EQUIVALENCIA DE ESTADOS:"]
        for estado in afd.estados:
            marcas = []
            if estado == afd.inicio:
                marcas.append("INICIAL")
            if estado in afd.aceptacion:
                marcas.append("ACEPTACIÓN")

            marca_txt = f" [{' / '.join(marcas)}]" if marcas else ""
            lineas.append(
                f"{afd.nombres[estado]} = "
                f"{formatear_conjunto(estado)}{marca_txt}"
            )

        lineas.append("")
        lineas.append("TRANSICIONES:")

        for origen in afd.estados:
            for simbolo in afd.alfabeto:
                destino = afd.transiciones[origen][simbolo]
                lineas.append(
                    f"{afd.nombres[origen]} --{simbolo}--> "
                    f"{afd.nombres[destino]}"
                )

        return "\n".join(lineas)

    def _posiciones_circulares(self, nombres, width, height):
        n = len(nombres)
        cx = width / 2
        cy = height / 2
        rx = max(120, min(width * 0.38, 360))
        ry = max(90, min(height * 0.34, 220))

        posiciones = {}
        if n == 1:
            posiciones[nombres[0]] = (cx, cy)
            return posiciones

        for i, nombre in enumerate(nombres):
            ang = -math.pi / 2 + 2 * math.pi * i / n
            x = cx + rx * math.cos(ang)
            y = cy + ry * math.sin(ang)
            posiciones[nombre] = (x, y)

        return posiciones

    def _flecha(self, canvas, x1, y1, x2, y2, etiqueta, curvatura=0):
        dx = x2 - x1
        dy = y2 - y1
        dist = max(math.hypot(dx, dy), 1)

        r = 29
        sx = x1 + dx / dist * r
        sy = y1 + dy / dist * r
        ex = x2 - dx / dist * r
        ey = y2 - dy / dist * r

        if curvatura:
            nx = -dy / dist
            ny = dx / dist
            mx = (sx + ex) / 2 + nx * curvatura
            my = (sy + ey) / 2 + ny * curvatura

            canvas.create_line(
                sx, sy, mx, my, ex, ey,
                fill="#64748b",
                width=2,
                arrow="last",
                smooth=True,
            )
            lx = mx + nx * 10
            ly = my + ny * 10
        else:
            canvas.create_line(
                sx, sy, ex, ey,
                fill="#64748b",
                width=2,
                arrow="last",
            )
            lx = (sx + ex) / 2
            ly = (sy + ey) / 2 - 10

        canvas.create_text(
            lx, ly,
            text=etiqueta,
            fill="#0f172a",
            font=("Arial", 10, "bold"),
        )

    def _bucle(self, canvas, x, y, etiqueta):
        canvas.create_arc(
            x - 22, y - 54,
            x + 22, y - 10,
            start=25,
            extent=300,
            style="arc",
            width=2,
            outline="#64748b",
        )
        canvas.create_text(
            x, y - 62,
            text=etiqueta,
            fill="#0f172a",
            font=("Arial", 10, "bold")
        )

    def _dibujar_nodo(self, canvas, x, y, nombre, inicial=False, aceptacion=False):
        r = 28
        fill = "#e0e7ff" if inicial else "#ffffff"
        outline = "#4f46e5" if inicial else "#334155"

        canvas.create_oval(
            x-r, y-r, x+r, y+r,
            fill=fill,
            outline=outline,
            width=3 if inicial else 2,
        )

        if aceptacion:
            canvas.create_oval(
                x-r+5, y-r+5, x+r-5, y+r-5,
                outline="#16a34a",
                width=2,
            )

        canvas.create_text(
            x, y,
            text=nombre,
            fill="#111827",
            font=("Arial", 11, "bold"),
        )

        if inicial:
            canvas.create_line(
                x-r-52, y,
                x-r-4, y,
                fill="#4f46e5",
                width=2,
                arrow="last",
            )

    def _dibujar_afn(self, afn):
        canvas = self.canvas_afn
        canvas.delete("all")
        canvas.update_idletasks()

        width = max(canvas.winfo_width(), 700)
        height = max(canvas.winfo_height(), 300)

        nombres = [f"q{x}" for x in sorted(afn.estados)]
        pos = self._posiciones_circulares(nombres, width, height)

        # Agrupar etiquetas por par origen-destino
        aristas = {}
        for origen in sorted(afn.estados):
            for simbolo, destinos in afn.transiciones.get(origen, {}).items():
                for destino in destinos:
                    aristas.setdefault(
                        (f"q{origen}", f"q{destino}"), []
                    ).append(simbolo)

        for (origen, destino), simbolos in aristas.items():
            x1, y1 = pos[origen]
            x2, y2 = pos[destino]
            etiqueta = ",".join(simbolos)

            if origen == destino:
                self._bucle(canvas, x1, y1, etiqueta)
            else:
                self._flecha(
                    canvas, x1, y1, x2, y2,
                    etiqueta,
                    curvatura=12
                )

        for estado in sorted(afn.estados):
            nombre = f"q{estado}"
            x, y = pos[nombre]
            self._dibujar_nodo(
                canvas,
                x, y,
                nombre,
                inicial=(estado == afn.inicio),
                aceptacion=(estado == afn.aceptacion),
            )

    def _dibujar_afd(self, afd):
        canvas = self.canvas_afd
        canvas.delete("all")
        canvas.update_idletasks()

        width = max(canvas.winfo_width(), 700)
        height = max(canvas.winfo_height(), 300)

        nombres = [afd.nombres[e] for e in afd.estados]
        pos = self._posiciones_circulares(nombres, width, height)

        # Agrupar símbolos que lleven entre el mismo par de estados.
        aristas = {}
        for origen in afd.estados:
            o = afd.nombres[origen]
            for simbolo, destino in afd.transiciones[origen].items():
                d = afd.nombres[destino]
                aristas.setdefault((o, d), []).append(simbolo)

        for (origen, destino), simbolos in aristas.items():
            x1, y1 = pos[origen]
            x2, y2 = pos[destino]
            etiqueta = ",".join(simbolos)

            if origen == destino:
                self._bucle(canvas, x1, y1, etiqueta)
            else:
                self._flecha(
                    canvas, x1, y1, x2, y2,
                    etiqueta,
                    curvatura=12
                )

        for estado in afd.estados:
            nombre = afd.nombres[estado]
            x, y = pos[nombre]
            self._dibujar_nodo(
                canvas,
                x, y,
                nombre,
                inicial=(estado == afd.inicio),
                aceptacion=(estado in afd.aceptacion),
            )


def main_gui():
    app = AplicacionAutomatas()
    app.mainloop()


if __name__ == "__main__":
    main_gui()
