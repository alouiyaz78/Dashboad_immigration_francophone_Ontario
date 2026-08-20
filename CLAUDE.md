## Agent skills

### Issue tracker

Issues live in GitHub Issues (`alouiyaz78/Dashboad_immigration_francophone_Ontario`), managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context layout (root `CONTEXT.md` + `docs/adr/`, created lazily as needed). See `docs/agents/domain.md`.

## Norme de design (permanente)

Toute interface ou visualisation ajoutée ou modifiée dans ce dashboard doit respecter les principes de Gestalt suivants. Cette règle s'applique à toute future modification (nouveaux graphiques, nouveaux composants d'interface), pas seulement à la revision de design documentee dans l'historique de commits.

- **Proximité** : regrouper visuellement les elements lies (espacement, sections) ; separer clairement ce qui ne l'est pas.
- **Similarite** : des elements de meme nature/role partagent un style visuel coherent (memes couleurs, memes formes) ; ne pas reutiliser un style visuel pour des elements de nature differente (ex: un style de badge/bouton ne doit pas etre applique a du texte non cliquable).
- **Hierarchie visuelle** : l'element le plus important a l'ecran doit etre le plus proeminent visuellement ; les elements secondaires (etiquettes, legendes, texte d'appoint) restent visuellement en retrait.
- **Non-redondance de l'encodage couleur** : la couleur ne doit jamais encoder deux fois la meme information qu'une dimension deja presente (hauteur, position, taille). Si une valeur est deja lisible via la position/taille, la couleur encode une autre variable ou reste neutre.
- **Priorite a l'element actionnable** : dans un composant interactif (filtre, bouton, lien), l'element interactif/actionnable doit toujours etre visuellement plus proeminent que son etiquette/label descriptif — jamais l'inverse.
