# ucas - BSD make (FreeBSD make or bmake)

PROG=	ucas
SRCS=	main.c util.c lex.c node.c parse.c eval.c analyze.c gates.c \
	find.c model.c remez.c approx.c nlp.c gif.c anim.c spirv.c sat.c glslc.c \
	plot.c
# calc: loops, closed forms, calculus, evaluation schemes
SRCS+=	loop.c cnum.c closed.c integ.c ode.c scheme.c cancel.c
# complex, polar and trig forms; asin acos atan2 sinh cosh tanh ...
SRCS+=	forms.c
# properties: monotonic, period, identify, zeros, ...
SRCS+=	props.c
# calc2: derivatives and their conditions, limits, series, Wronskians
SRCS+=	calc2.c
# symbolic algebra: polynomials, rational functions, integration, solve
SRCS+=	poly.c alg.c rint.c solve.c
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
nlp_words.h: nlp.c nlp_calc2.inc nlp_alg.inc nlp_words.awk
	awk -f ${.CURDIR}/src/nlp_words.awk ${.CURDIR}/src/nlp.c \
	    ${.CURDIR}/src/nlp_calc2.inc ${.CURDIR}/src/nlp_alg.inc > ${.TARGET}
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

accept: ${PROG}
	@for t in ${TESTS}; do \
		n=$${t%.ucas}; \
		(cd ${.CURDIR}/tests && ${.OBJDIR}/${PROG} $$t) > \
		    ${.CURDIR}/tests/$$n.out 2>&1; \
		echo "wrote tests/$$n.out"; \
	done

CLEANFILES+=	*.got *.diff

# Optional Discord front end (README, "Discord bot"): its own Makefile in
# discord/, needs libcurl.  Never built by `make` or `make test`.
.PHONY: discord discord-test
discord:
	cd ${.CURDIR}/discord && ${MAKE}
discord-test:
	cd ${.CURDIR}/discord && ${MAKE} test

.include <bsd.prog.mk>
