# The words nlp.c knows: every word of four or more letters inside its
# string literals, once each, as C string initialisers.  Misspelt words in
# English input are corrected towards these (nlp.c, spellfix).
# (comment lines are skipped: they may quote misspellings)
/^[ \t]*(\/\*|\*)/ { next }
{
	line = $0
	while (match(line, /"([^"\\]|\\.)*"/)) {
		s = substr(line, RSTART + 1, RLENGTH - 2)
		line = substr(line, RSTART + RLENGTH)
		gsub(/\\./, " ", s)
		n = split(tolower(s), w, /[^a-z-]+/)
		for (i = 1; i <= n; i++) {
			x = w[i]
			gsub(/^-+|-+$/, "", x)
			if (length(x) >= 4 && !(x in seen)) {
				seen[x] = 1
				print "\t\"" x "\","
			}
		}
	}
}
