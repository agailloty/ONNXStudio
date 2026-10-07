# Vérification d’ONNXStudioUI — 7 octobre 2026

Références : `../ONNXStudio/specs/architecture/ONNXStudio-Modular-Monolith.md`,
`User-Controls-Strategy.md`, `Component-Communication-Patterns.md` et les six
user stories de `for_architect/`.

## Architecture conservée

- Un processus : Avalonia héberge Kestrel ; UI et API partagent le registre et les services Core.
- Core reste indépendant d’Avalonia et d’ASP.NET Core.
- ViewModels MVVM, commandes CommunityToolkit, bindings compilés et ViewLocator sans réflexion.
- Les schémas JSON, exemples et document OpenAPI appartiennent au module API.
- Le graphe est un contrôle Avalonia réutilisable alimenté par propriétés et commande de sélection.

Les versions déjà employées par le dépôt (.NET 10 et Avalonia 12) sont conservées.
Les documents citent également .NET 8 et Avalonia 11 : cela ne justifie pas une rétrogradation.

## Corrections et vérifications

| Parcours | Résultat |
| --- | --- |
| US-001, chargement | Sélection multiple depuis l’accueil et le dashboard ; Ctrl+O/Cmd+O et dépôt de plusieurs fichiers gérés par la fenêtre ; chargements sérialisés ; annulation transmise au loader ; arrêt du lot sur erreur ou annulation ; retour depuis une erreur réparé. |
| US-002, inspection | Remplacement de la liste d’opérations par un graphe avec arêtes, couleurs, sélection, zoom centré sur le pointeur, déplacement, navigation par flèches et remise à zéro avec Home ; rendu limité au viewport. Recherche, catégories, attributs, statistiques et métadonnées restent disponibles. |
| US-003, formulaires | Validation du rang, des dimensions fixes et du nombre d’éléments ; inférence d’une dimension dynamique ; saisie explicite des formes ambiguës ; précision Int64 conservée ; rejet des fractions pour les entiers et des nombres non finis ; champs texte exécutables ; réinitialisation. |
| US-004, inférence | Exécution hors du thread UI ; absence de résultats périmés après une erreur ; scalaires de rang zéro ; entrées/sorties texte, booléennes et numériques ; indices des cinq meilleures classes conservés avec scores bruts. |
| Cycle de vie ONNX | Éviction sur déchargement ; une session encore utilisée est libérée à la fin de l’inférence ; suppression des écrans mis en cache et libération des clients HTTP/abonnements. |
| US-005, API | Démarrage après chargement ; arrêt au déchargement du dernier modèle ; démarrages/arrêts sérialisés ; port 0 fonctionnel ; nettoyage après échec de démarrage ; état synchronisé entre écrans ; erreurs de port affichées ; schémas/exemples respectant les formes et noms JSON ; `/openapi.json` et informations à `/`. |
| US-006, sandbox | Endpoints GET et POST sélectionnables ; vrais appels HTTP ; validation JSON ; réponses formatées, en-têtes, statut coloré, durée ; historique de dix requêtes par endpoint avec relance ; export cURL ; erreurs sans réutiliser un ancien statut 200. |
| Paramètres | Thème sombre par défaut ; sauvegarde atomique du thème et du port dans LocalApplicationData/ONNXStudio/settings.json ; repli sur les valeurs par défaut si le fichier est absent ou corrompu. |
| Arrêt application | Disposition asynchrone du conteneur, nécessaire pour fermer le serveur embarqué. |

## Validation reproductible

```powershell
dotnet restore ONNXStudio.slnx
dotnet build ONNXStudio.slnx -c Release --no-restore -p:UsedAvaloniaProducts=
dotnet test ONNXStudio.slnx -c Release --no-restore -p:UsedAvaloniaProducts=
```

`UsedAvaloniaProducts=` désactive uniquement la tâche de télémétrie de compilation,
qui tente d’écrire hors du workspace dans cet environnement restreint.

Les tests utilisent de vrais modèles ONNX et un vrai serveur Kestrel sur port local
éphémère. Les tests Avalonia Headless chargent les écrans et exercent la sélection,
le zoom, le déplacement et la navigation clavier du graphe. Ils couvrent aussi le
parcours chargement → inspection → inférence → HTTP → déchargement, les conflits
de port, les sessions actives lors d’une éviction et la persistance des paramètres.

Résultat sous Windows : **80 tests réussis** (51 Core/API existants et 29 nouveaux
tests UI/intégration), compilation Release sans erreur ni avertissement.

## Écarts restant à traiter avant une conformité exhaustive

La compilation et les tests ne constituent pas une certification de toutes les
fonctionnalités décrites dans les spécifications.

- **Inspection avancée** : repli des sous-graphes, histogrammes/valeurs des poids,
  exploration détaillée des tenseurs intermédiaires et accessibilité complète du graphe.
- **Formulaires** : grille matricielle, aperçu d’image, choix du prétraitement,
  tokenisation NLP associée au modèle, sauvegarde de jeux d’entrées/defaults par modèle,
  rendu des sorties images et export des résultats. Les images utilisent actuellement
  CHW et une normalisation 0–1 ; les tenseurs numériques attendent des valeurs déjà préparées.
- **API** : Swagger UI, contrat de requête groupant plusieurs objets indépendants,
  personnalisation des noms de champs, configuration CORS dans l’UI et exports Python/PowerShell.
  Les batchs de tenseurs passent par les dimensions du modèle. Le sandbox refuse les réponses
  dépassant 10 Mio plutôt que de les tronquer.
- **Chargement** : les modèles exigeant des opérateurs personnalisés non enregistrés
  restent rejetés par ONNX Runtime ; la validation native en cours ne peut pas être
  interrompue immédiatement, même si l’annulation empêche son enregistrement.
- **Production** : publication NativeAOT/trimming non validée ; l’API comporte encore
  des contrats sérialisés dynamiquement. Distribution et interactions natives à tester
  sur macOS/Linux, ainsi que les performances/mémoire sur ResNet/BERT et les graphes
  de 1 000 à 10 000 nœuds. La journalisation persistante complète reste à configurer.

Les documents sources ont été conservés sans modification ; leurs cases de validation
n’ont pas été cochées sur la seule base de ces tests.
