# ucas - BSD make (FreeBSD make or bmake)

PROG=	ucas
SRCS=	main.c util.c lex.c node.c parse.c eval.c analyze.c gates.c \
	find.c model.c remez.c approx.c nlp.c gif.c anim.c spirv.c sat.c glslc.c \
	plot.c fsimp.c nodesat.c bound.c
# calc: loops, closed forms, calculus, evaluation schemes
SRCS+=	loop.c cnum.c closed.c integ.c ode.c scheme.c cancel.c
# complex, polar and trig forms; asin acos atan2 sinh cosh tanh ...
SRCS+=	forms.c
# properties: monotonic, period, identify, zeros, ...
SRCS+=	props.c
# calc2: derivatives and their conditions, limits, series, Wronskians
SRCS+=	calc2.c
# complex, quaternion, octonion and Cayley-Dickson numbers
SRCS+=	hyper.c
# symbolic algebra: polynomials, rational functions, integration, solve
SRCS+=	poly.c alg.c rint.c solve.c
# rings and finite fields: Z/n, GF(p^n), polynomials and matrices over them
SRCS+=	ntheory.c gfield.c ring.c ring_nlp.c
ring.o ring.pieo: ring_zn.inc ring_gf.inc ring_misc.inc ring_cmd.inc
# absalg: exact abstract algebra (absalg.h): integers, integer matrices,
# abelian groups, permutation groups, Groebner bases
SRCS+=	zint.c zmat.c fgab.c pgrp.c ideal.c absalg_cmd.c
# viz: integrals shaded, Riemann sums, solids of revolution (plot's renderer)
SRCS+=	viz.c
# transforms (Laplace, Z, Fourier, DFT), series and infinite sums
SRCS+=	xfrat.c xform.c series.c
# the JSON frontend: a JSON Lines protocol, structured results
SRCS+=	json.c report.c jsonmode.c
# an MCP server over stdio, on top of the JSON frontend
SRCS+=	mcp.c
# proofs: an LCF kernel, its checker, decision procedures, a theory library
SRCS+=	kterm.c knorm.c kernel.c kcheck.c kauto.c kscript.c prove_cmd.c \
	prove_model.c
thy_absalg.inc: thy_absalg.thy
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' \
	    ${.ALLSRC} > ${.TARGET}
kscript.o kscript.pieo: thy_absalg.inc
CLEANFILES+=	thy_absalg.inc
# homological algebra: spaces, (co)homology, exact couples (homalg.h)
SRCS+=	simp.c cx.c excouple.c homalg_cmd.c
# simplify's function-level pass over a shader's GLSL source
SRCS+=	glslfn.c
# ScarletNX's GLSL ES compiler (src/snxglslc, kept as it is there): its
# parser gives glslfn.c the syntax tree.  Its glslc.c would clash with
# ours as an object, so its files build into a library of their own.
SNXGLSLC=	lexer parser emit builtins glslc
SNXDIR=	${.CURDIR}/src/snxglslc
.for f in ${SNXGLSLC}
snx_${f}.o: ${SNXDIR}/${f}.c ${SNXDIR}/glslc.h ${SNXDIR}/internal.h
	${CC} ${CFLAGS} -fPIE -c ${SNXDIR}/${f}.c -o ${.TARGET}
.endfor
libsnxglslc.a: ${SNXGLSLC:S/^/snx_/:S/$/.o/}
	rm -f ${.TARGET}
	${AR} rcs ${.TARGET} ${.ALLSRC}
LDADD+=	libsnxglslc.a
DPADD+=	libsnxglslc.a
${PROG}.full ${PROG}: libsnxglslc.a
CLEANFILES+=	libsnxglslc.a ${SNXGLSLC:S/^/snx_/:S/$/.o/}
glslfn.o glslfn.pieo: ${SNXDIR}/glslc.h ${SNXDIR}/internal.h
.PATH:	${.CURDIR}/src
MAN=

CSTD=	c99
CFLAGS+=	-O2 -I${.CURDIR}/src
CWARNFLAGS+=	-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare \
		-Wno-missing-field-initializers
WARNS?=	2
MK_WERROR=	no
LDADD+=	-lm

# nlp.c corrects misspelt words towards the words of its own strings
CFLAGS+=	-I${.OBJDIR}
nlp_words.h: nlp.c nlp_calc2.inc nlp_hyper.inc nlp_alg.inc nlp_xform.inc \
    nlp_words.awk
	awk -f ${.CURDIR}/src/nlp_words.awk ${.CURDIR}/src/nlp.c \
	    ${.CURDIR}/src/nlp_calc2.inc ${.CURDIR}/src/nlp_hyper.inc \
	    ${.CURDIR}/src/nlp_alg.inc ${.CURDIR}/src/nlp_xform.inc > ${.TARGET}
nlp.o nlp.pieo: nlp_words.h
CLEANFILES+=	nlp_words.h

TESTS!=	cd ${.CURDIR}/tests && ls *.ucas 2>/dev/null || true

# Each tests/NAME.ucas runs and must print exactly tests/NAME.out.
# `make accept` rewrites the .out files from the current output.
test: ${PROG}
	@${.OBJDIR}/${PROG} --sat-selftest || exit 1
	@fail=0; for t in ${TESTS}; do \
		n=$${t%.ucas}; \
		(cd ${.CURDIR}/tests && ${.OBJDIR}/${PROG} $$t) > $$n.got 2>&1; \
		if diff -u ${.CURDIR}/tests/$$n.out $$n.got > $$n.diff; then \
			echo "ok   $$n"; rm -f $$n.got $$n.diff; \
		else echo "FAIL $$n (see $$n.diff)"; fail=1; fi; \
	done; exit $$fail
	@cd ${.CURDIR} && ${MAKE} json-test
	@cd ${.CURDIR} && ${MAKE} mcp-test

accept: ${PROG}
	@for t in ${TESTS}; do \
		n=$${t%.ucas}; \
		(cd ${.CURDIR}/tests && ${.OBJDIR}/${PROG} $$t) > \
		    ${.CURDIR}/tests/$$n.out 2>&1; \
		echo "wrote tests/$$n.out"; \
	done
	@cd ${.CURDIR} && ${MAKE} json-accept
	@cd ${.CURDIR} && ${MAKE} mcp-accept

CLEANFILES+=	*.got *.diff

# The JSON frontend: tests/json/NAME.jsonl, one request per line, must
# answer exactly tests/json/NAME.out (files inline or not written).
JTESTS!=	cd ${.CURDIR}/tests/json && ls *.jsonl 2>/dev/null || true

json-test: ${PROG}
	@fail=0; for t in ${JTESTS}; do \
		n=$${t%.jsonl}; \
		(cd ${.CURDIR}/tests/json && ${.OBJDIR}/${PROG} --json --files none \
		    < $$t) > json-$$n.got 2>&1; \
		if diff -u ${.CURDIR}/tests/json/$$n.out json-$$n.got > \
		    json-$$n.diff; then \
			echo "ok   json/$$n"; rm -f json-$$n.got json-$$n.diff; \
		else echo "FAIL json/$$n (see json-$$n.diff)"; fail=1; fi; \
	done; exit $$fail

json-accept: ${PROG}
	@for t in ${JTESTS}; do \
		n=$${t%.jsonl}; \
		(cd ${.CURDIR}/tests/json && ${.OBJDIR}/${PROG} --json --files none \
		    < $$t) > ${.CURDIR}/tests/json/$$n.out 2>&1; \
		echo "wrote tests/json/$$n.out"; \
	done

.PHONY: json-test json-accept

# The MCP server: tests/mcp/NAME.jsonl (JSON-RPC lines) must answer
# exactly tests/mcp/NAME.out; NAME.flags holds extra arguments (--flat).
MTESTS!=	cd ${.CURDIR}/tests/mcp && ls *.jsonl 2>/dev/null || true

mcp-test: ${PROG}
	@fail=0; for t in ${MTESTS}; do \
		n=$${t%.jsonl}; \
		fl=`cat ${.CURDIR}/tests/mcp/$$n.flags 2>/dev/null || true`; \
		(cd ${.CURDIR}/tests/mcp && ${.OBJDIR}/${PROG} --mcp $$fl \
		    < $$t) > mcp-$$n.got 2>&1; \
		if diff -u ${.CURDIR}/tests/mcp/$$n.out mcp-$$n.got > \
		    mcp-$$n.diff; then \
			echo "ok   mcp/$$n"; rm -f mcp-$$n.got mcp-$$n.diff; \
		else echo "FAIL mcp/$$n (see mcp-$$n.diff)"; fail=1; fi; \
	done; exit $$fail

mcp-accept: ${PROG}
	@for t in ${MTESTS}; do \
		n=$${t%.jsonl}; \
		fl=`cat ${.CURDIR}/tests/mcp/$$n.flags 2>/dev/null || true`; \
		(cd ${.CURDIR}/tests/mcp && ${.OBJDIR}/${PROG} --mcp $$fl \
		    < $$t) > ${.CURDIR}/tests/mcp/$$n.out 2>&1; \
		echo "wrote tests/mcp/$$n.out"; \
	done

.PHONY: mcp-test mcp-accept

# Optional Discord front end (README, "Discord bot"): its own Makefile in
# discord/, needs libcurl.  Never built by `make` or `make test`.
.PHONY: discord discord-test
discord:
	cd ${.CURDIR}/discord && ${MAKE}
discord-test:
	cd ${.CURDIR}/discord && ${MAKE} test

.include <bsd.prog.mk>
