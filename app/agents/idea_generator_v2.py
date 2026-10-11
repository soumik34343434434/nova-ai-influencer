"""High-volume, diversity-aware idea generation for NOVA.

The generator creates many cheap structured concepts first, then lets the jury
decide what deserves expensive script/animation work. It intentionally does not
copy trend wording; trends become constraints (format, hook, emotion, pacing).

Every idea carries a scene contract the renderer can draw: one setting, one
emoji prop with its label, and the beats on which that prop is on screen. Lines
only name that prop, so the joke is visible on screen. Titles and captions are
built from curated words only, never from headline text.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import re
from typing import Any, Iterable, NamedTuple

from app.agents.action_packs import action_spec, renderable_characters, role_actions
from app.agents.cartoon_research import evergreen_topics
from app.agents.cartoon_safety import public_text_problems


@dataclass(frozen=True)
class IdeaCandidateV2:
    idea_id: str
    title: str
    hook: str
    character: str
    format_name: str
    trend_id: str
    emotion: str
    angle: str
    duration_target: float
    beats: tuple[dict[str, Any], ...]
    production_cost: float
    source_evidence: tuple[str, ...] = ()


CHARACTER_PERSONAS = {
    "MOMO": ("sleepy", "sarcastic", "cute"),
    "PIP": ("mischievous", "curious", "chaotic"),
    "BUBU": ("innocent", "eager", "overconfident"),
}

# Every idea is four beats: hook -> escalation -> reaction (the twist) ->
# punchline. The beat images come from the character's action pack
# (config/cartoon_action_packs.json), so an idea can only ask for actions that
# can actually be rendered. Lines use {C} (character) and {P} (the prop label).
# Lines never name anything else on screen, so a viewer can follow every joke.
#
# Each format has six premises, each a complete four-beat script. A premise is
# (hook, hook_pattern, goal, escalation, twist, punchline); the hook pattern tag
# lets the learning loop measure which opening styles work.
class Premise(NamedTuple):
    hook: str
    hook_pattern: str
    goal: str
    escalation: str
    twist: str
    punch: str


FORMAT_RECIPES: dict[str, dict[str, Any]] = {
    "dance": {
        "duration": 11.0,
        "angles": ("adorable_commitment", "wrong_move", "sudden_mood_change", "beat_drop_surprise", "overconfident_start"),
        "premises": (
            Premise("{C} learned one move with the {P}. Then the beat dropped!", "twist_tease",
                    "{C} plans one tiny dance step with the {P}.", "Then the song speeds up. The {P} dances too, fast!",
                    "Wait, why? Turns out the {P} only knows the slow version.", "{C} stares at the camera. Still dancing, apparently."),
            Premise("Nobody told {C} this was a dance battle with the {P}?!", "curiosity",
                    "{C} wants to show off one move to the {P}.", "Then the {P} copies every move, faster and faster!",
                    "Instead of winning, {C} trips over the {P}.", "{C}: of course. Still counts as a routine."),
            Premise("{C} starts one dance and the {P} joins in. Big mistake!", "warning",
                    "{C} spins one time, very carefully, next to the {P}.", "Then the {P} spins too, and never stops!",
                    "Instead, the {P} followed a different beat.", "{C}: still spinning, apparently. This is my personality now."),
            Premise("Picture this: {C} wants one calm dance with the {P}!", "pov",
                    "{C} starts one calm dance with the {P}.", "Then the music gets loud. The {P} gets wild!",
                    "Instead of calm, the {P} wants a full routine?", "{C}: of course. Same dance again, tomorrow."),
            Premise("{C} dances for one second. Then the {P} took over!", "twist_tease",
                    "{C} tries one graceful twirl with the {P}.", "Then the {P} bounces higher than {C}!",
                    "Wait, why? Turns out the {P} is the real dancer.", "{C}: apparently I am still only the backup dancer."),
            Premise("One tiny dance with the {P}. Watch {C} go wrong!", "countdown",
                    "{C} copies one move from the {P}.", "Then the music jumps. The {P} keeps dancing, unbothered!",
                    "Instead of matching, {C} does it completely wrong.", "{C}: still a dance, apparently."),
        ),
    },
    "trend_audio": {
        "duration": 11.0,
        "angles": ("audio_mismatch", "cute_reveal", "deadpan_switch", "beat_misread", "unexpected_pause"),
        "premises": (
            Premise("{C} heard one sound near the {P}. Then it changed!", "twist_tease",
                    "{C} wants to match one sound with the {P}.", "Then the audio glitches. The {P} remixes everything!",
                    "Wait, why? Turns out the {P} was playing it backwards.", "{C}: apparently that still counts as one take."),
            Premise("The sound says calm. Nobody warned {C} about the {P}?!", "curiosity",
                    "{C} tries one tiny lip-sync to the {P}.", "Then the sound jumps. The {P} sings much too loudly!",
                    "Instead of singing, {C} freezes like the {P} is listening.", "{C} stares at the camera. The song was still the warning."),
            Premise("{C} pressed play on the {P}. Big mistake!", "warning",
                    "{C} listens to one beat from the {P}.", "Then the beat drops. The {P} bounces everywhere!",
                    "Instead, the {P} had a different song in mind.", "{C}: of course. Still on beat, somehow."),
            Premise("Picture this: {C} finds a song that matches the {P} perfectly!", "pov",
                    "{C} tries one clean move to the {P}.", "Then the chorus hits. The {P} joins in, off beat!",
                    "Instead of a clean move, {C} tangles up with the {P}.", "{C}: apparently the chorus is still not mine."),
            Premise("One sound, one {P}, one terrible decision by {C}!", "countdown",
                    "{C} presses the {P} for one short clip.", "Then the {P} loops the clip again and again!",
                    "Wait, why? Turns out the loop was the whole song.", "{C}: still one take. Still on repeat."),
            Premise("{C} wanted a quiet sound. Then the {P} got loud!", "twist_tease",
                    "{C} keeps the {P} on one soft note.", "Then the {P} hums louder, and louder, and louder!",
                    "Instead of quiet, the {P} decided to become a drum solo.", "{C}: apparently quiet is still not on the playlist."),
        ),
    },
    "reaction": {
        "duration": 11.0,
        "angles": ("shock", "overconfidence", "deadpan", "tiny_detail_big_reaction", "wrong_notification"),
        "premises": (
            Premise("{C} touched the {P} for one second. Big mistake!", "warning",
                    "{C} just wants one normal look at the {P}.", "Then the {P} buzzes again. And again, and again!",
                    "Instead of news, the {P} says it's time to relax?", "{C}: I should have ignored that one."),
            Premise("{C} saw the {P} light up. Then {C} regretted it!", "twist_tease",
                    "{C} thought one look at the {P} was harmless.", "Then the {P} lights up again. The reactions get bigger!",
                    "Instead, the message on the {P} was just a sticker.", "{C}: of course. Still not relaxing."),
            Premise("Nobody warned {C} that the {P} could be this dramatic?", "curiosity",
                    "{C} gives the {P} one calm, polite tap.", "Then the {P} shakes and {C} jumps back!",
                    "Instead of calm, the {P} made a huge surprise noise.", "{C} stares at the camera. Calm was still a lie."),
            Premise("Picture this: {C} holds the {P} and it wiggles. Big mistake!", "pov",
                    "{C} wants to hold the {P} for one second.", "Then the {P} wiggles. {C} panics!",
                    "Instead of holding, {C} drops the {P} by accident.", "{C}: of course. Back to normal, apparently not."),
            Premise("{C} is calm about the {P}. For about two seconds!", "countdown",
                    "{C} tries one calm nod at the {P}.", "Then the {P} flashes bright. {C} gasps!",
                    "Instead, the {P} was only saying hello.", "{C}: apparently hello is still a problem."),
            Premise("{C} spotted one harmless {P}. Then it got worse!", "twist_tease",
                    "{C} reads one tiny line on the {P}.", "Then the {P} lights up. {C} jumps in surprise!",
                    "Instead of reading, {C} tosses the {P} into the air?", "{C}: I should have ignored that one."),
        ),
    },
    "pov": {
        "duration": 11.0,
        "angles": ("relatable_work", "social_anxiety", "small_problem_big_reaction", "monday_mood", "quiet_minute"),
        "premises": (
            Premise("Picture this: {C} finally gets one quiet minute with the {P}!", "pov",
                    "{C} sits down with the {P} for one quiet minute.", "Then the {P} lights up. Tiny emergency, again!",
                    "Wait, why? Turns out the quiet minute was the emergency.", "{C}: apparently the universe still disagrees."),
            Premise("Picture this: {C} had one simple job with the {P}?!", "pov",
                    "{C} has one easy job: keep the {P} safe.", "Then the {P} wobbles. Problem number two!",
                    "Instead of safe, the {P} rolls away and {C} chases it.", "{C}: of course. Back to work, again."),
            Premise("Picture this: the {P} asks {C} for one favor. Big mistake!", "warning",
                    "{C} agrees to one small favor for the {P}.", "Then the favor grows. The {P} asks for more!",
                    "Instead of one favor, {C} gets a whole list.", "{C}: apparently one favor was never one favor."),
            Premise("Picture this: {C} is relaxing when the {P} starts talking?", "pov",
                    "{C} wants one lazy moment with the {P}.", "Then the {P} gets louder, and {C} gets sleepier!",
                    "Instead, the {P} wanted a nap buddy.", "{C}: of course. Nap time is still never really nap time."),
            Premise("Picture this: {C} tries one normal morning with the {P}!", "pov",
                    "{C} tries one slow start to the morning with the {P}.", "Then the {P} rings again and again!",
                    "Instead of waking up, {C} sleeps right through the {P}.", "{C} stares at the camera. Still the same morning."),
            Premise("Picture this: {C} wants a break, but the {P} disagrees!", "pov",
                    "{C} takes one calm break with the {P}.", "Then the {P} starts spinning and {C} is dizzy!",
                    "Instead of a break, the {P} turned it into a workout?", "{C}: apparently breaks are still not in the plan."),
        ),
    },
    "story_comedy": {
        "duration": 12.0,
        "angles": ("escalation", "misunderstanding", "reverse_expectation", "false_win", "loop_back"),
        "premises": (
            Premise("{C} thought today was easy with the {P}. Big mistake!", "warning",
                    "{C} has one simple plan for the {P}.", "Then the problem multiplies. The {P} keeps going!",
                    "Instead of fixing it, the {P} misunderstood everything?", "{C}: of course. Still one job, apparently."),
            Premise("One tiny problem with the {P}. Then {C} helped!", "twist_tease",
                    "{C} tries to fix one tiny thing with the {P}.", "Then it gets worse. The {P} fights back!",
                    "Actually, why? The {P} had other plans.", "{C}: put it back. All of it."),
            Premise("Nobody told {C} the {P} had a secret plan?!", "curiosity",
                    "{C} picks up the {P} for one simple task.", "Then the {P} grows. More and more of it!",
                    "Instead, the {P} was never on {C}'s side.", "{C} stares at the camera. I still lost, apparently."),
            Premise("{C} asked the {P} for one favor. Big mistake!", "warning",
                    "{C} wants one easy result from the {P}.", "Then the {P} takes over the whole plan!",
                    "Instead of easy, the {P} made it a complicated mess.", "{C}: apparently easy is still not on the menu."),
            Premise("It was one normal day for {C}, until the {P} arrived!", "twist_tease",
                    "{C} tries one ordinary routine with the {P}.", "Then the {P} wobbles and everything goes wrong!",
                    "Wait, why? Turns out the routine was the wrong one.", "{C}: still counts as a routine, I guess."),
            Premise("{C} wanted one tidy moment with the {P}. Then chaos!", "twist_tease",
                    "{C} tidies up one spot with the {P}.", "Then the {P} rolls everywhere. Chaos, again!",
                    "Instead of tidy, the {P} misunderstood the whole cleanup.", "{C}: put it back. All of it."),
        ),
    },
    "challenge": {
        "duration": 11.0,
        "angles": ("attempts", "countdown", "cheat_then_fail", "almost_win", "rules_change"),
        "premises": (
            Premise("{C} said yes to the {P} before the rules. Big mistake!", "warning",
                    "{C} accepts one easy challenge with the {P}.", "Then the rules change. The {P} adds another round!",
                    "Wait, why? Turns out the prize was another challenge.", "{C}: apparently that's still not in the tutorial."),
            Premise("One simple challenge with the {P}. Can {C} survive it?", "challenge",
                    "{C} tries one attempt with the {P}. Easy, right?", "Then the {P} changes the target. Harder now!",
                    "Instead of winning, {C} gets stuck to the {P}.", "{C}: apparently I still lost round one."),
            Premise("{C} is holding the {P} for a whole minute. Watch closely!", "countdown",
                    "{C} counts to three, one step at a time, holding the {P}.", "Then the {P} gets heavier. Way heavier!",
                    "Instead of three, {C} keeps counting forever?", "{C}: of course. Still counting."),
            Premise("{C} has to balance the {P} on one foot. No help!", "challenge",
                    "{C} lifts one foot and balances the {P}.", "Then the {P} wobbles. Problem number two!",
                    "Instead, the {P} was always the one balancing {C}.", "{C} stares at the camera. The foot is still not mine."),
            Premise("{C} cheated at the {P} game. Then the rules changed!", "twist_tease",
                    "{C} takes one shortcut with the {P}.", "Then the {P} bounces back and forth. Round after round!",
                    "Instead of a shortcut, {C} took the long way around the {P}.", "{C}: apparently shortcuts are still not allowed."),
            Premise("Three tries, one {P}, and {C} is still not done!", "list",
                    "{C} takes the first try with the {P}.", "Then the {P} is moved again. Another try!",
                    "Instead of finishing, {C} keeps going because the rules changed.", "{C}: still not finished, apparently."),
        ),
    },
    "transformation": {
        "duration": 11.0,
        "angles": ("upgrade_backfire", "before_after", "expectation_reality", "wrong_upgrade", "side_effect"),
        "premises": (
            Premise("{C} upgraded the {P} one tiny bit. Big mistake!", "warning",
                    "{C} wants to improve one small part of the {P}.", "Then the upgrade spreads. The {P} changes everywhere!",
                    "Instead of better, the {P} misunderstood the whole upgrade?", "{C}: put it back. Put it all back."),
            Premise("Before: the {P} is perfect. Then {C} touched it!", "before_after",
                    "{C} plans one small upgrade for the {P}.", "Then the {P} starts glowing. Too bright now!",
                    "Instead, the glow was the wrong upgrade.", "{C}: apparently perfect was still the upgrade."),
            Premise("{C} made the {P} twice as fancy in one second!", "twist_tease",
                    "{C} adds one shiny piece to the {P}.", "Then the shiny piece grows. Bigger and bigger!",
                    "Instead of shiny, the {P} turned into something unexpected.", "{C} stares at the camera. Shiny was still not the goal."),
            Premise("{C} wants a new look for the {P}. Watch the change!", "countdown",
                    "{C} picks one bright color for the {P}.", "Then the color spreads. The {P} is covered!",
                    "Instead of a new look, the {P} got a whole new shape?", "{C}: of course. Same {P}, different problem."),
            Premise("After: a brand new {P}. Before: nobody asked for that!", "before_after",
                    "{C} hopes one tiny upgrade will fix the {P}.", "Then the {P} keeps changing. Again and again!",
                    "Instead, the new version does the opposite of helping.", "{C}: apparently the old one was still better."),
            Premise("{C} finally fixes the {P}. Or does it?", "curiosity",
                    "{C} gives the {P} one clever adjustment.", "Then the adjustment backfires. The {P} spins!",
                    "Instead of fixed, the {P} is now a different shape?", "{C}: still counts as an upgrade. Apparently."),
        ),
    },
    "cute_animal": {
        "duration": 11.0,
        "angles": ("tiny_hero", "adorable_failure", "comfort", "protective", "tiny_solution_big_effect"),
        "premises": (
            Premise("{C} is trying to help with the {P}. Then it happened!", "twist_tease",
                    "{C} wants to help with one tiny thing: the {P}.", "Then the tiny plan grows. The {P} is everywhere now!",
                    "Wait, why? Turns out the help was literally the problem.", "{C}: still counts as helping. Apparently."),
            Premise("Nobody asked {C} to help with the {P}. {C} helped anyway!", "curiosity",
                    "{C} has one adorable plan for the {P}.", "Then the {P} rolls off the table. Oops!",
                    "Instead of helping, {C} hugs the {P} too hard.", "{C}: of course. Helping is still a full time job."),
            Premise("{C} saw one sad {P}. Big mistake!", "warning",
                    "{C} gives the {P} one tiny, cozy blanket.", "Then the {P} wiggles free. {C} chases it!",
                    "Instead of cozy, the {P} looks more confused than before?", "{C} stares at the camera. Still trying, though."),
            Premise("{C} wants to protect the {P} from everything. Everything!", "curiosity",
                    "{C} stands guard next to one {P}.", "Then a tiny noise starts. {C} jumps up!",
                    "Instead, the noise was the {P} snoring.", "{C}: apparently guarding is still a nap job."),
            Premise("{C} is tiny and the {P} is huge. Big plan!", "twist_tease",
                    "{C} pushes the {P} with one cute little push.", "Then the {P} doesn't move. Then it moves too much!",
                    "Instead of moving, the {P} rolls right over {C}?", "{C}: apparently tiny is still not the same as strong."),
            Premise("{C} has one adorable idea for the {P}. Watch!", "countdown",
                    "{C} tucks the {P} in for one cozy nap.", "Then the {P} starts bouncing. Nap is over!",
                    "Instead of sleeping, {C} is now bouncing too.", "{C}: still cozy, apparently."),
        ),
    },
    "absurdist": {
        "duration": 11.0,
        "angles": ("logic_break", "escalation", "deadpan_acceptance", "reality_glitch", "nonsense_rule"),
        "premises": (
            Premise("{C} accepted this way too quickly. Then the {P} quit!", "twist_tease",
                    "{C} just wants one normal minute with the {P}.", "Then gravity takes a break. The {P} floats up!",
                    "Instead of falling, the {P} starts giving advice?", "{C}: of course. Same time tomorrow, again."),
            Premise("This should not be possible. Nobody told {C} about the {P}?", "curiosity",
                    "{C} tries one ordinary task with the {P}.", "Then the {P} starts spinning in place, backwards!",
                    "Instead, the {P} is following its own rules.", "{C} stares at the camera. Apparently that's still a rule."),
            Premise("{C} asked the {P} for one normal answer. Big mistake!", "warning",
                    "{C} asks the {P} one ordinary question.", "Then the {P} answers in a song. Loudly!",
                    "Instead of an answer, the {P} wants a dance?", "{C}: apparently nonsense is still the answer."),
            Premise("Physics quit for one second, and {C} noticed the {P}!", "twist_tease",
                    "{C} tries one steady hold on the {P}.", "Then the {P} bounces up. Then sideways, then up again!",
                    "Instead of steady, {C} floats too?", "{C}: still standing, apparently."),
            Premise("{C} has a perfectly normal plan with the {P}. Watch!", "countdown",
                    "{C} follows one simple step with the {P}.", "Then the step turns into a loop. The {P} loops forever!",
                    "Instead, the loop was the step all along.", "{C}: apparently loops still count as progress."),
            Premise("Nobody explained the rules to {C} and the {P}?!", "curiosity",
                    "{C} reads one rule on the {P} out loud.", "Then the rules flip upside down. The {P} flips too!",
                    "Instead of rules, the {P} starts making new ones.", "{C}: of course. The rules are still wrong."),
        ),
    },
    "listicle": {
        "duration": 12.0,
        "angles": ("three_failures", "three_reactions", "countdown", "top_bad_choices", "one_more"),
        "premises": (
            Premise("Three things {C} should never do with the {P}!", "list",
                    "{C} tries number three: one tiny poke at the {P}.", "Then number two happens. The {P} gets worse!",
                    "Wait, why? Turns out number one was the instructions.", "{C}: apparently the list still wins. Again."),
            Premise("{C}'s top three bad decisions with the {P}. Big mistake!", "list",
                    "{C} picks one harmless idea for the {P}.", "Then decision two backfires. The {P} bounces everywhere!",
                    "Instead of harmless, the {P} turned into the worst idea?", "{C}: apparently number three was still the {P}."),
            Premise("Three reasons the {P} is trouble. Reason one is {C}!", "list",
                    "{C} gives reason one a big smile at the {P}.", "Then reason two shows up. The {P} is spinning!",
                    "Instead of reason three, the {P} ran off with the list?", "{C} stares at the camera. Still reason one, apparently."),
            Premise("Top three ways {C} wrecks a simple {P} plan. Number one!", "list",
                    "{C} starts the plan with one careful tap on the {P}.", "Then way two: the {P} spins out of control!",
                    "Instead, way three was just {C} giving up.", "{C}: apparently the plan is still the problem."),
            Premise("Three ways to lose to the {P}. {C} tries all three!", "countdown",
                    "{C} takes one first try with the {P}.", "Then way two: the {P} rolls faster and faster!",
                    "Instead of way three, {C} gives the {P} a big hug?", "{C}: apparently hugs still count as losing."),
            Premise("Three tiny steps for {C}, one giant mess with the {P}!", "list",
                    "{C} takes step one with the {P}.", "Then step two: the {P} grows a bit. Then a lot!",
                    "Instead of step three, the {P} took a break?", "{C}: still on step one, apparently."),
        ),
    },
    "micro_short": {
        "duration": 10.0,
        "angles": ("single_gag", "instant_reversal", "reaction_loop", "setup_payoff_echo", "micro_twist"),
        "premises": (
            Premise("{C} had one job with the {P}. Then it moved!", "warning",
                    "{C} has one job: keep the {P} still.", "Then it breaks. The {P} is in pieces!",
                    "Instead of fixing it, {C} misunderstood the instructions?", "{C}: of course. Still one job."),
            Premise("Seven seconds to go wrong with the {P}. Watch {C}!", "countdown",
                    "{C} finishes one tiny task with the {P}.", "Then the {P} bounces twice. Oops!",
                    "Wait, why? Turns out the task was the wrong one.", "{C} stares at the camera. Seven seconds, still counting."),
            Premise("One quick tap on the {P}, and {C} is in trouble!", "twist_tease",
                    "{C} taps the {P} one time, very gently.", "Then the {P} rolls off. Everything goes wrong!",
                    "Instead of gentle, the tap was a giant push?", "{C}: apparently one tap was enough."),
            Premise("{C} wanted to hold the {P} for one second. Big mistake!", "warning",
                    "{C} holds one {P} with both paws.", "Then the {P} wiggles. Loose again!",
                    "Instead of holding, {C} is holding nothing now?", "{C}: of course. Back to one job."),
            Premise("{C} is ready for the {P} trick. Watch the reversal!", "twist_tease",
                    "{C} does one trick with the {P}.", "Then the trick reverses. The {P} comes back!",
                    "Instead of finishing, {C} tries the trick backwards?", "{C}: still one trick. Still one job."),
            Premise("Nobody gave {C} a second chance with the {P}. Oops!", "curiosity",
                    "{C} gets one more chance to hold the {P}.", "Then the {P} escapes again. Faster this time!",
                    "Instead of a second chance, {C} gets a rerun?", "{C}: apparently one job means many jobs."),
        ),
    },
}

# Each scene prop: emoji drawn by the renderer, the label spoken in lines, and the set
# the renderer draws behind the character. Keys are internal; never shown to viewers.
PROPS: dict[str, tuple[str, str, str]] = {
    "sandwich": ("🥪", "sandwich", "kitchen"),
    "phone": ("📱", "phone", "bedroom"),
    "book": ("📚", "book", "classroom"),
    "ball": ("⚽", "ball", "park"),
    "cupcake": ("🧁", "cupcake", "bakery"),
    "teddy": ("🧸", "teddy bear", "living_room"),
    "sunflower": ("🌻", "sunflower", "garden"),
    "laptop": ("💻", "laptop", "studio"),
}

# Trend keywords that suggest a prop. Only these words are read from research text;
# no headline words ever reach a title, caption or line.
PROP_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("phone", "phone"), ("meme", "phone"), ("notification", "phone"), ("app", "phone"),
    ("homework", "book"), ("school", "book"), ("exam", "book"), ("study", "book"),
    ("food", "sandwich"), ("snack", "sandwich"), ("kitchen", "sandwich"), ("recipe", "sandwich"),
    ("cake", "cupcake"), ("bakery", "cupcake"), ("dessert", "cupcake"),
    ("cat", "teddy"), ("cats", "teddy"), ("pet", "teddy"), ("puppy", "teddy"), ("kitten", "teddy"),
    ("garden", "sunflower"), ("plant", "sunflower"), ("flower", "sunflower"), ("rain", "sunflower"),
    ("robot", "laptop"), ("gadget", "laptop"), ("ai", "laptop"), ("laptop", "laptop"),
    ("office", "laptop"), ("tech", "laptop"),
    ("ball", "ball"), ("cricket", "ball"), ("football", "ball"), ("sport", "ball"), ("park", "ball"),
)

# Distinct sound cues per beat role (all synthesised by cartoon_audio.make_sfx).
ROLE_SFX = {
    "hook": ("notification", "pop", "whoosh", "tap"),
    "escalation": ("impact", "chaos"),
    "reaction": ("overwhelmed", "reaction_hit"),
    "punchline": ("punchline",),
}

FORMAT_HOOK_SFX = {"dance": "beat", "trend_audio": "beat", "challenge": "countdown", "listicle": "countdown"}
BEAT_ROLES = ("hook", "escalation", "reaction", "punchline")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:70]


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("||".join(parts).encode()).hexdigest(), 16)


def _prop_key(text: str, *seed: str) -> str:
    words = set(re.findall(r"[a-z0-9']+", (text or "").lower()))
    for keyword, key in PROP_KEYWORDS:
        if keyword in words:
            return key
    keys = tuple(PROPS)
    return keys[_seed(*seed, "prop") % len(keys)]


def _angle_phrase(angle: str) -> str:
    return {
        "adorable_commitment": "commits too hard",
        "wrong_move": "does it completely wrong",
        "sudden_mood_change": "loses the mood instantly",
        "audio_mismatch": "the sound does not match reality",
        "cute_reveal": "the cute part has a catch",
        "deadpan_switch": "goes deadpan at the worst moment",
        "shock": "gets surprised by a tiny detail",
        "overconfidence": "is way too confident",
        "deadpan": "pretends nothing happened",
        "relatable_work": "tries to protect one quiet minute",
        "social_anxiety": "avoids one awkward moment",
        "small_problem_big_reaction": "turns a tiny problem into chaos",
        "escalation": "lets the small problem grow",
        "misunderstanding": "misreads the situation",
        "reverse_expectation": "gets the opposite result",
        "attempts": "keeps trying after the warning",
        "countdown": "runs out of patience",
        "cheat_then_fail": "takes a shortcut that backfires",
        "upgrade_backfire": "upgrades the wrong thing",
        "before_after": "gets a very different after",
        "expectation_reality": "gets the exact opposite",
        "tiny_hero": "tries to save the day",
        "adorable_failure": "fails in the cutest way",
        "comfort": "turns chaos into comfort",
        "logic_break": "accepts impossible logic",
        "deadpan_acceptance": "acts like this is normal",
        "three_failures": "collects three bad decisions",
        "three_reactions": "has three increasingly worse reactions",
        "single_gag": "has exactly one job",
        "instant_reversal": "wins for one second then loses",
        "reaction_loop": "reacts to the same problem twice",
    }.get(angle, angle.replace("_", " "))


def _pick(options: tuple[Any, ...], *key: str) -> Any:
    return options[_seed(*key) % len(options)]


def _fill(template: str, character: str, prop_label: str) -> str:
    text = template.format(C=character, P=prop_label)
    # A line can open a new sentence after a full stop; keep it capitalised.
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def _make_beats(character: str, fmt: str, premise: Premise, prop_label: str, emotion: str,
                key: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    """Build the four renderable beats for one idea, or () if the character has no complete pack."""
    roles = role_actions(character)
    if not roles:
        return ()
    lines = {
        "hook": _fill(premise.goal, character, prop_label),
        "escalation": _fill(premise.escalation, character, prop_label),
        "reaction": _fill(premise.twist, character, prop_label),
        "punchline": _fill(premise.punch, character, prop_label),
    }
    hook_sfx = FORMAT_HOOK_SFX.get(fmt) or _pick(ROLE_SFX["hook"], *key, "sfx")
    beats = []
    for role in BEAT_ROLES:
        action = roles[role]
        spec = action_spec(character, action) or {}
        words = len(re.findall(r"[A-Za-z0-9']+", lines[role]))
        beats.append({
            "beat": role,
            "asset_action": action,
            "voice": lines[role],
            "visual": str(spec.get("visual", "")),
            "emotion": str(spec.get("emotion", emotion)),
            "framing": str(spec.get("framing", "medium")),
            "sfx": hook_sfx if role == "hook" else _pick(ROLE_SFX[role], *key, role),
            "duration": round(max(2.0, words / 2.8 + 0.6), 2),
        })
    return tuple(beats)


def scene_contract(prop_key: str, beats: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """The renderer's contract: prop and setting, plus the beats whose lines name the prop."""
    emoji, label, setting = PROPS[prop_key]
    prop_beats = [i for i, beat in enumerate(beats) if label in str(beat.get("voice", "")).lower()]
    return {"setting": setting, "prop": emoji, "prop_label": label, "prop_beats": prop_beats}


def fallback_trend_rows(recent_titles: Iterable[str] = (), count: int = 6) -> list[dict[str, Any]]:
    """Safe, date-rotated premises used when no live trend survives research."""
    rows = []
    for topic in evergreen_topics(recent_titles=list(recent_titles), count=count):
        rows.append({
            "trend_id": f"evergreen-{_slug(str(topic['title']))}",
            "title": str(topic["title"]),
            "category": str(topic["category"]),
            "formats": tuple(topic["formats"]),
            "emotions": tuple(topic["emotions"]),
        })
    return rows


def generate_ideas(trends: Iterable[dict[str, Any]], formats: Iterable[dict[str, Any]] | None = None,
                   count: int = 60, characters: tuple[str, ...] = ("MOMO", "PIP", "BUBU")) -> list[dict[str, Any]]:
    trend_rows = list(trends)
    format_rows = list(formats or [])
    if not trend_rows:
        trend_rows = fallback_trend_rows()
    # Asset capability gate: only cast characters whose action pack covers
    # every beat role, so no generated idea is unrenderable.
    cast = renderable_characters(characters)

    candidates: list[dict[str, Any]] = []
    for trend in trend_rows[:20]:
        trend_id = str(trend.get("trend_id", ""))
        trend_formats = list(trend.get("formats") or ())
        if format_rows:
            trend_formats.extend(str(x.get("name")) for x in format_rows[:8] if x.get("name"))
        trend_formats.extend(["micro_short", "reaction", "story_comedy", "absurdist", "cute_animal"])
        unique_formats = list(dict.fromkeys(x for x in trend_formats if x in FORMAT_RECIPES))
        if not unique_formats:
            unique_formats = ["micro_short"]

        emotions = list(trend.get("emotions") or ()) or ["amusement", "surprise", "cute"]
        title_source = str(trend.get("title", ""))
        for fmt in unique_formats[:8]:
            recipe = FORMAT_RECIPES[fmt]
            for character in cast:
                for angle in recipe["angles"]:
                    key = (trend_id, fmt, character, angle)
                    emotion = emotions[_seed(trend_id, fmt, character, angle) % len(emotions)]
                    premise_index = _seed(*key, "premise") % len(recipe["premises"])
                    premise = recipe["premises"][premise_index]
                    prop_key = _prop_key(title_source, *key)
                    prop_label = PROPS[prop_key][1]
                    title = f"{character} and the {prop_label}: {_angle_phrase(angle)}"
                    hook = _fill(premise.hook, character, prop_label)
                    beats = _make_beats(character, fmt, premise, prop_label, emotion, key)
                    if not beats:
                        continue
                    # Fail closed: public text must pass the family-safety filter.
                    if public_text_problems(title, hook):
                        continue
                    idea_id = f"idea-{_slug(title)}-{abs(_seed(title, trend_id)) % 100000:05d}"
                    cost = 0.25 + 0.08 * len(beats) + 0.06 * (1 if fmt in {"dance", "trend_audio"} else 0)
                    candidates.append({
                        "idea_id": idea_id,
                        "title": title,
                        "hook": hook,
                        "hook_pattern": premise.hook_pattern,
                        "premise_id": f"{fmt}-{premise_index}",
                        "character": character,
                        "format_name": fmt,
                        "trend_id": trend_id,
                        "emotion": emotion,
                        "angle": angle,
                        "duration_target": float(recipe["duration"]),
                        "beats": list(beats),
                        "scene": scene_contract(prop_key, beats),
                        "production_cost": round(min(cost, 1.0), 3),
                        "source_evidence": list(trend.get("evidence", ())),
                        "trend_score": float(trend.get("opportunity_score", trend.get("score", 0.0)) or 0.0),
                        "saturation_score": float(trend.get("saturation_score", 0.0) or 0.0),
                        "velocity_score": float(trend.get("velocity_score", 0.0) or 0.0),
                    })
    # Deterministic near-duplicate suppression. We prefer different formats,
    # characters and emotional angles before keeping another semantically similar idea.
    candidates.sort(key=lambda x: (
        -x.get("trend_score", 0.0),
        x["production_cost"],
        _seed(x["idea_id"]),
    ))
    chosen: list[dict[str, Any]] = []
    seen_tokens: list[set[str]] = []
    for item in candidates:
        toks = set(re.findall(r"[a-z0-9]+", (item["title"] + " " + item["hook"]).lower()))
        duplicate = False
        for old, old_tokens in zip(chosen, seen_tokens):
            union = toks | old_tokens
            if union and len(toks & old_tokens) / len(union) >= 0.72:
                # Keep diversity even when wording is similar.
                if item["format_name"] == old["format_name"] and item["character"] == old["character"]:
                    duplicate = True
                    break
        if duplicate:
            continue
        chosen.append(item)
        seen_tokens.append(toks)
        if len(chosen) >= count:
            break
    return chosen


def idea_to_dict(item: IdeaCandidateV2 | dict[str, Any]) -> dict[str, Any]:
    return asdict(item) if isinstance(item, IdeaCandidateV2) else dict(item)
