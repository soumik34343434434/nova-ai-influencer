# NOVA Visual World

The recurring world is intentionally structured as data so every episode can reuse the same character rules, locations, wardrobe families and prop vocabulary.

Scenes:
- office — modern Indian tech office; workplace and meeting comedy
- home — minimalist apartment desk at night; productivity and internet observations
- cafe — modern Indian cafe; creator/lifestyle and experiment moments
- metro — modern Indian metro carriage; commute and social observations

The project stores scene briefs in `config/world.json`. The storyboard engine selects a scene, wardrobe and primary prop for every shot.

Reference images were generated during the setup pass with the canonical NOVA character reference. They are not committed as production files because provider-hosted image URLs are temporary; regenerate them through the media-provider layer when a hosted image backend is enabled. The local renderer remains functional without them.
