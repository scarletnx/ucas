# ucas-discord - the optional Discord front end.  BSD make.
#
#   make            fetch and build Concord, then ucas-discord
#   make test       unit checks and fake gateway events (no network)
#
# Needs libcurl (pkg curl) and, for Concord's own build, git and gmake.
# The core ucas binary (../Makefile) does not depend on any of this.

CONCORD_REPO?=	https://github.com/Cogmasters/concord
CONCORD_REV?=	fb0f834d07d65048a6afde177e318fe9659a9494	# v2.4.0
CONCORD=	${.CURDIR}/third_party/concord
CONCORD_LIB=	${CONCORD}/lib/libdiscord.a
LOCALBASE?=	/usr/local

PROG=	ucas-discord
# every engine source except the CLI's main.c, so new engine files
# (and the commands in them) come along without edits here
ENGINE!=	cd ${.CURDIR}/../src && ls *.c | grep -v '^main\.c$$'
SRCS=	bot.c attach.c sup.c text.c selftest.c botuse.c dcjson.c ${ENGINE}
.PATH:	${.CURDIR}/../src
MAN=

# nlp.c includes a word list generated from its own strings
CFLAGS+=	-I${.OBJDIR}
nlp_words.h: nlp.c nlp_calc2.inc nlp_hyper.inc nlp_alg.inc nlp_xform.inc \
    nlp_img.inc nlp_words.awk
	awk -f ${.CURDIR}/../src/nlp_words.awk ${.CURDIR}/../src/nlp.c \
	    ${.CURDIR}/../src/nlp_calc2.inc ${.CURDIR}/../src/nlp_hyper.inc \
	    ${.CURDIR}/../src/nlp_alg.inc ${.CURDIR}/../src/nlp_xform.inc \
	    ${.CURDIR}/../src/nlp_img.inc > ${.TARGET}
nlp.o nlp.pieo: nlp_words.h
CLEANFILES+=	nlp_words.h
# the proof library (prove_cmd.c), as ../Makefile embeds it
thy_absalg.inc: thy_absalg.thy
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' \
	    ${.ALLSRC} > ${.TARGET}
kscript.o kscript.pieo: thy_absalg.inc
CLEANFILES+=	thy_absalg.inc
# the help for bots ("explain AI usage"), from its Markdown
.for f in ai_usage ai_usage_short
${f}.inc: ${f}.md
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' \
	    ${.ALLSRC} > ${.TARGET}
CLEANFILES+=	${f}.inc
.endfor
botuse.o botuse.pieo: ai_usage.inc ai_usage_short.inc

# img.c compiles stb_image (../src/third_party, unmodified): its warnings
# are not ours
img.o img.pieo: img.c img.h third_party/stb_image.h
	${CC} ${CFLAGS:N-W*} -w -fPIE -c ${.CURDIR}/../src/img.c -o ${.TARGET}

CSTD=	c99
CFLAGS+=	-O2 -I${.CURDIR}/../src -I${.CURDIR}
CFLAGS+=	-isystem ${CONCORD}/include -isystem ${CONCORD}/core \
		-isystem ${CONCORD}/gencodecs -isystem ${LOCALBASE}/include
CWARNFLAGS+=	-Wall -Wextra -Wno-unused-parameter -Wno-sign-compare \
		-Wno-missing-field-initializers
WARNS?=	2
MK_WERROR=	no
LDADD+=	${CONCORD_LIB} -L${LOCALBASE}/lib -lcurl -lpthread -lm
DPADD+=	${CONCORD_LIB}

# ScarletNX's GLSL ES compiler (../src/snxglslc), as ../Makefile builds
# it: a library of its own, as its glslc.c would clash with ours
SNXGLSLC=	lexer parser emit builtins glslc
SNXDIR=	${.CURDIR}/../src/snxglslc
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

# gmake gets no MAKEFLAGS: under make -j they carry bmake's -J, which it
# refuses
${CONCORD_LIB}:
	@if [ ! -d ${CONCORD}/.git ]; then \
		mkdir -p ${.CURDIR}/third_party && \
		git clone -q ${CONCORD_REPO} ${CONCORD} && \
		git -C ${CONCORD} checkout -q ${CONCORD_REV}; \
	fi
	cd ${CONCORD} && env MAKEFLAGS= \
	    CFLAGS="-O2 -fPIC -w -I${LOCALBASE}/include" \
	    gmake -s CC=${CC} static >/dev/null

# Each tests/NAME.jsonl is a list of gateway events; --stdin must print
# exactly tests/NAME.out.  A short CPU limit keeps the timeout case
# quick (the wall clock is long, so a busy host still hits CPU first);
# tests/NAME.env, when present, replaces them.  spirv-cross (simplify's
# GLSL view) is off unless NAME.env sets UCAS_SPIRV_CROSS; a test that
# needs a tool names it in NAME.needs and is skipped without it.
TESTENV=	UCAS_DISCORD_CPU=2 UCAS_DISCORD_WALL=60 UCAS_DISCORD_MAXOUT=65536
TESTPRE=	e="${TESTENV}"; [ -f ${.CURDIR}/tests/$$n.env ] && \
		    e=$$(cat ${.CURDIR}/tests/$$n.env); \
		if [ -f ${.CURDIR}/tests/$$n.needs ] && ! command -v \
		    $$(cat ${.CURDIR}/tests/$$n.needs) >/dev/null 2>&1; then \
			echo "skip discord/$$n (no $$(cat ${.CURDIR}/tests/$$n.needs))"; \
			continue; \
		fi
DTESTS!=	cd ${.CURDIR}/tests && ls *.jsonl 2>/dev/null || true

test: ${PROG}
	@${.OBJDIR}/${PROG} --selftest
	@fail=0; for t in ${DTESTS}; do \
		n=$${t%.jsonl}; ${TESTPRE}; \
		env UCAS_SPIRV_CROSS=none $$e ${.OBJDIR}/${PROG} --stdin \
		    < ${.CURDIR}/tests/$$t > $$n.got 2>&1; \
		if diff -u ${.CURDIR}/tests/$$n.out $$n.got > $$n.diff; then \
			echo "ok   discord/$$n"; rm -f $$n.got $$n.diff; \
		else echo "FAIL discord/$$n (see $$n.diff)"; fail=1; fi; \
	done; exit $$fail

accept: ${PROG}
	@for t in ${DTESTS}; do \
		n=$${t%.jsonl}; ${TESTPRE}; \
		env UCAS_SPIRV_CROSS=none $$e ${.OBJDIR}/${PROG} --stdin \
		    < ${.CURDIR}/tests/$$t > ${.CURDIR}/tests/$$n.out 2>&1; \
		echo "wrote discord/tests/$$n.out"; \
	done

CLEANFILES+=	*.got *.diff

.include <bsd.prog.mk>

# Concord's generated headers must exist before anything includes them
${OBJS}: ${CONCORD_LIB}
