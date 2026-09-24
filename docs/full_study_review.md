# Full participant-flow review links

The researcher-only `POST /api/study/admin/review-link` endpoint accepts a
`domain` and issues a signed, release-bound invitation for group A. A link
is valid for 30 days and permits six independent browser sessions. The
invitation appears in a URL fragment, is retained in session storage and
removed from the displayed URL; access logs never contain it. It grants no
access to administrator APIs or existing participant sessions.

`/review/<domain>/` loads the same consent renderer, client assets, tutorials,
Task 1/2/3 flow, survey, reward policy, explanation protocol and configured
model service as the paid experiment. Only assignment/accounting differs:
group A is fixed, the session is labelled as an unpaid researcher preview,
and the final page does not issue a real Prolific completion code.

Review instances have `mode=preview`, an internal `review-` participant ID,
and no Prolific identity link. Formal exports must select `mode=pilot` (or
the study's linked participant cohort). Review records cannot consume paid
places or enter the payment report. Separate HttpOnly cookies for each game
allow the three review links to coexist without replacing a real participant
cookie or each other's progress.

No GET request creates an enrollment. After consent, the invite is verified
and its capacity is checked inside the serialized enrollment transaction.
The client cannot select another domain, group or recruitment mode. The
normal participant API remains unchanged.
