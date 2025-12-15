# ============================================
# VOICI LE CODE CORRIGÉ - Copiez TOUT depuis ici :
# ============================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import linprog
import warnings
import datetime
warnings.filterwarnings('ignore')

# ============================================
# 1. CLASSES AVEC INTERPRÉTATION DE COHÉRENCE
# ============================================

class FuzzyTriangular:
    def __init__(self, l, m, u):
        self.l = l
        self.m = m
        self.u = u
    
    def __repr__(self):
        return f"({self.l}, {self.m}, {self.u})"
    
    def alpha_cut(self, alpha):
        return (self.l + alpha*(self.m-self.l), 
                self.u - alpha*(self.u-self.m))

class FuzzyBWM_Solver:
    def __init__(self):
        self.criteria = []
        self.best_criterion = None
        self.worst_criterion = None
        self.BO_matrix = []
        self.OW_matrix = []
        self.weights = None
        self.consistency = None
        self.consistency_ratio = None
        
        # Échelle floue étendue
        self.fuzzy_scales = {
            '1': FuzzyTriangular(1, 1, 1),      # Égal
            '2': FuzzyTriangular(1, 2, 3),      # Faible
            '3': FuzzyTriangular(2, 3, 4),      # Modéré
            '4': FuzzyTriangular(3, 4, 5),      # Modéré+
            '5': FuzzyTriangular(4, 5, 6),      # Fort
            '6': FuzzyTriangular(5, 6, 7),      # Fort+
            '7': FuzzyTriangular(6, 7, 8),      # Très fort
            '8': FuzzyTriangular(7, 8, 9),      # Très fort+
            '9': FuzzyTriangular(8, 9, 9)       # Extrême
        }
        
        # Dictionnaire pour les labels
        self.scale_labels = {
            '1': 'Égal',
            '2': 'Faible', 
            '3': 'Modéré',
            '4': 'Modéré+',
            '5': 'Fort',
            '6': 'Fort+',
            '7': 'Très fort',
            '8': 'Très fort+',
            '9': 'Extrême'
        }
    
    def add_criteria(self, criteria_list):
        self.criteria = criteria_list
    
    def set_best_worst(self, best, worst):
        self.best_criterion = best
        self.worst_criterion = worst
    
    def set_comparisons(self, BO_comparisons, OW_comparisons):
        self.BO_matrix = BO_comparisons
        self.OW_matrix = OW_comparisons
    
    def solve(self, alpha=0.5):
        n = len(self.criteria)
        
        if not self.BO_matrix or not self.OW_matrix:
            return None, None, None
        
        if len(self.BO_matrix) != n or len(self.OW_matrix) != n:
            st.error("Erreur: matrices de taille incorrecte!")
            return None, None, None
        
        try:
            weights, consistency = self._solve_fuzzy_bwm(alpha)
            self.weights = weights
            self.consistency = consistency
            self.consistency_ratio = self._calculate_consistency_ratio(consistency)
            return weights, consistency, self.consistency_ratio
        except Exception as e:
            st.error(f"Erreur: {e}")
            return None, None, None
    
    def _solve_fuzzy_bwm(self, alpha):
        n = len(self.criteria)
        
        # Alpha-cuts
        BO_intervals = []
        for fuzzy in self.BO_matrix:
            lower, upper = fuzzy.alpha_cut(alpha)
            BO_intervals.append((lower, upper))
        
        OW_intervals = []
        for fuzzy in self.OW_matrix:
            lower, upper = fuzzy.alpha_cut(alpha)
            OW_intervals.append((lower, upper))
        
        # Optimisation
        c = [1] + [0] * n
        
        A_eq = [[0] + [1] * n]
        b_eq = [1]
        
        A_ub = []
        b_ub = []
        
        # Contraintes BO
        for j in range(n):
            A_ub.append([-1] + [0] * n)
            A_ub[-1][j+1] = -BO_intervals[j][0]
            b_ub.append(0)
            
            A_ub.append([-1] + [0] * n)
            A_ub[-1][j+1] = BO_intervals[j][1]
            b_ub.append(0)
        
        worst_idx = self.criteria.index(self.worst_criterion)
        
        # Contraintes OW
        for j in range(n):
            A_ub.append([-1] + [0] * n)
            A_ub[-1][j+1] = 1
            A_ub[-1][worst_idx+1] = -OW_intervals[j][0]
            b_ub.append(0)
            
            A_ub.append([-1] + [0] * n)
            A_ub[-1][j+1] = -1
            A_ub[-1][worst_idx+1] = OW_intervals[j][1]
            b_ub.append(0)
        
        bounds = [(0, None)] + [(0, 1)] * n
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')
        
        if res.success:
            weights = res.x[1:]
            consistency = res.x[0]
            weights = weights / np.sum(weights)
            return weights, consistency
        else:
            raise ValueError("Optimisation échouée")
    
    def _calculate_consistency_ratio(self, xi):
        """Calcule le ratio de cohérence selon Guo & Zhao (2017)"""
        n = len(self.criteria)
        
        # Table des indices de cohérence (CI)
        ci_table = {
            1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63,
            5: 2.30, 6: 3.00, 7: 3.73, 8: 4.47,
            9: 5.23
        }
        
        CI = ci_table.get(n, 2.5*n - 4)  # Formule approximative si n > 9
        
        if CI == 0:
            return 0.0
        
        return xi / CI
    
    def get_consistency_interpretation(self, xi, cr):
        """Retourne l'interprétation détaillée de la cohérence"""
        
        interpretations = []
        
        # Interprétation basée sur ξ
        if xi < 0.1:
            xi_status = "✅ EXCELLENTE"
            xi_message = "Vos jugements sont très cohérents"
        elif xi < 0.2:
            xi_status = "👍 BONNE"
            xi_message = "Vos jugements sont acceptables"
        elif xi < 0.3:
            xi_status = "⚠️ MOYENNE"
            xi_message = "Considérez revoir certaines comparaisons"
        else:
            xi_status = "❌ FAIBLE"
            xi_message = "Vos jugements sont incohérents"
        
        interpretations.append(f"**Indice ξ = {xi:.4f}** - {xi_status}")
        interpretations.append(f"*{xi_message}*")
        
        # Interprétation basée sur CR
        if cr < 0.1:
            cr_status = "✅ TRÈS BON"
            cr_message = "Cohérence satisfaisante"
            color = "green"
        elif cr < 0.2:
            cr_status = "👍 ACCEPTABLE"
            cr_message = "Cohérence acceptable pour la prise de décision"
            color = "orange"
        else:
            cr_status = "❌ INACCEPTABLE"
            cr_message = "Revisez vos comparaisons pour améliorer la cohérence"
            color = "red"
        
        interpretations.append(f"\n**Ratio CR = {cr:.3f}** - {cr_status}")
        interpretations.append(f"<span style='color:{color}'>{cr_message}</span>")
        
        # Explication théorique
        theory = """
        **Théorie de la cohérence Fuzzy BWM:**
        
        L'indice ξ mesure l'écart maximal entre:
        1. w_B / w_j ≈ ã_Bj  (Best vs Others)
        2. w_j / w_W ≈ ã_jW  (Others vs Worst)
        
        Pour une cohérence parfaite, on devrait avoir:
        ã_Bj × ã_jW ≈ ã_BW
        
        **Interprétation pratique:**
        - ξ < 0.1 : Jugements très cohérents
        - 0.1 ≤ ξ < 0.2 : Jugements acceptables  
        - 0.2 ≤ ξ < 0.3 : Jugements à vérifier
        - ξ ≥ 0.3 : Jugements incohérents
        
        **Ratio de cohérence (CR):**
        CR = ξ / CI  où CI est l'indice de cohérence aléatoire
        - CR < 0.1 : Très bonne cohérence
        - 0.1 ≤ CR < 0.2 : Cohérence acceptable
        - CR ≥ 0.2 : Cohérence inacceptable
        """
        
        interpretations.append(f"\n{theory}")
        
        # Suggestions d'amélioration si nécessaire
        if cr >= 0.2:
            suggestions = """
            **Suggestions pour améliorer la cohérence:**
            
            1. **Vérifiez la relation:** ã_Bj × ã_jW ≈ ã_BW
            2. **Assurez-vous que:** Si A > B et B > C, alors A > C
            3. **Utilisez une progression logique** entre les valeurs
            4. **Meilleur vs Pire** doit être la valeur la plus élevée (8 ou 9)
            5. **Les autres valeurs** doivent être intermédiaires
            """
            interpretations.append(suggestions)
        
        return "\n\n".join(interpretations)
    
    def check_specific_inconsistencies(self):
        """Détecte les incohérences spécifiques"""
        inconsistencies = []
        
        best_idx = self.criteria.index(self.best_criterion)
        worst_idx = self.criteria.index(self.worst_criterion)
        
        # Récupérer ã_BW
        a_BW = self.BO_matrix[worst_idx]
        
        for j in range(len(self.criteria)):
            if j != best_idx and j != worst_idx:
                a_Bj = self.BO_matrix[j]
                a_jW = self.OW_matrix[j]
                
                # Calcul approximatif du produit
                prod_l = a_Bj.l * a_jW.l
                prod_u = a_Bj.u * a_jW.u
                
                # Vérifier si a_BW est dans l'intervalle produit
                if not (prod_l <= a_BW.u and prod_u >= a_BW.l):
                    inconsistencies.append({
                        'critere': self.criteria[j],
                        'a_Bj': str(a_Bj),
                        'a_jW': str(a_jW),
                        'produit_approx': f"({prod_l:.1f}, {prod_u:.1f})",
                        'a_BW': str(a_BW),
                        'probleme': f"a_Bj × a_jW ≠ a_BW"
                    })
        
        return inconsistencies

# ============================================
# 2. INTERFACE UTILISATEUR COMPLÈTE
# ============================================

def main():
    st.set_page_config(
        page_title="Fuzzy BWM Personnalisé - Production H₂ Vert",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé
    st.markdown("""
    <style>
    .main-title {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .criteria-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #5fba7d;
        margin: 1rem 0;
    }
    .consistency-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ffc107;
    }
    .bad-consistency {
        background: #f8d7da;
        border: 2px solid #dc3545;
    }
    .good-consistency {
        background: #d4edda;
        border: 2px solid #28a745;
    }
    .scale-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    .scale-table th, .scale-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }
    .scale-table th {
        background-color: #1e3c72;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Titre principal
    st.markdown("""
    <div class="main-title">
        <h1>🌍 FUZZY BWM PERSONNALISÉ</h1>
        <h3>Production d'Hydrogène Vert au Maroc - Interface Complète</h3>
        <p>Définissez vos propres critères, évaluez et analysez la cohérence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation du solveur
    if 'solver' not in st.session_state:
        st.session_state.solver = FuzzyBWM_Solver()
    
    solver = st.session_state.solver
    
    # Sidebar avec échelle floue
    with st.sidebar:
        st.markdown("### 📊 Échelle Floue de Saaty")
        
        # Table de l'échelle
        scale_data = {
            "Valeur": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "Signification": [
                "Égal", "Faible", "Modéré", "Modéré+", 
                "Fort", "Fort+", "Très fort", "Très fort+", "Extrême"
            ],
            "Nombre Flou": [
                "(1,1,1)", "(1,2,3)", "(2,3,4)", "(3,4,5)",
                "(4,5,6)", "(5,6,7)", "(6,7,8)", "(7,8,9)", "(8,9,9)"
            ]
        }
        
        st.table(pd.DataFrame(scale_data))
        
        st.markdown("---")
        st.markdown("### ⚙️ Paramètres")
        
        alpha = st.slider(
            "Niveau de confiance α:",
            0.0, 1.0, 0.5, 0.1,
            help="Niveau de coupe pour la défuzzification (0=conservateur, 1=optimiste)"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ Guide")
        with st.expander("Comment utiliser:"):
            st.markdown("""
            1. **Étape 1:** Entrez vos critères d'évaluation
            2. **Étape 2:** Sélectionnez Best et Worst
            3. **Étape 3:** Remplissez les matrices de comparaison
            4. **Étape 4:** Analysez la cohérence et les résultats
            5. **Étape 5:** Téléchargez les résultats
            """)
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 1. Définition des Critères", 
        "⚖️ 2. Comparaisons Floues", 
        "📊 3. Résultats et Cohérence", 
        "💾 4. Export des Résultats"
    ])
    
    # ============================================
    # TAB 1: DÉFINITION DES CRITÈRES PERSONNALISÉS
    # ============================================
    with tab1:
        st.markdown("### 🎯 Étape 1: Définissez vos critères d'évaluation")
        
        # Option 1: Utiliser des critères prédéfinis
        # Option 2: Entrer ses propres critères
        
        option = st.radio(
            "Choisissez votre mode d'entrée:",
            ["📋 Utiliser des critères prédéfinis", "✏️ Entrer mes propres critères"],
            horizontal=True
        )
        
        if option == "📋 Utiliser des critères prédéfinis":
            # Sélection du domaine
            domain = st.selectbox(
                "Domaine d'application:",
                ["Énergie Solaire (PV)", "Énergie Éolienne", "Production H₂", "Général"]
            )
            
            # Critères par domaine
            if domain == "Énergie Solaire (PV)":
                default_criteria = [
                    "Rayonnement solaire (GHI - kWh/m²/an)",
                    "Température moyenne annuelle (°C)",
                    "Pente du terrain (%)",
                    "Distance aux zones urbaines (km)",
                    "Distance aux routes principales (km)",
                    "Proximité des sources d'eau (km)",
                    "Distance au réseau électrique (km)",
                    "Coût du terrain (DH/m²)",
                    "Impact environnemental"
                ]
            elif domain == "Énergie Éolienne":
                default_criteria = [
                    "Vitesse moyenne du vent (m/s)",
                    "Densité de l'air",
                    "Turbulence (%)",
                    "Hauteur du mât disponible (m)",
                    "Distance aux zones habitées (km)",
                    "Accès aux routes (km)",
                    "Proximité réseau électrique (km)",
                    "Régularité des vents",
                    "Risques naturels"
                ]
            elif domain == "Production H₂":
                default_criteria = [
                    "Disponibilité en eau (m³/jour)",
                    "Qualité de l'eau (pH, minéraux)",
                    "Énergie disponible (MW)",
                    "Coût de l'électricité (DH/kWh)",
                    "Infrastructure existante",
                    "Marché de l'H₂ local",
                    "Subventions disponibles",
                    "Réglementations",
                    "Acceptation sociale"
                ]
            else:  # Général
                default_criteria = [
                    "Critère économique",
                    "Critère technique",
                    "Critère environnemental",
                    "Critère social",
                    "Critère de faisabilité",
                    "Critère temporel",
                    "Critère de risque"
                ]
            
            # Éditeur avec valeurs par défaut
            criteria_input = st.text_area(
                "Modifiez la liste des critères si nécessaire (un par ligne):",
                value="\n".join(default_criteria),
                height=200,
                help="Un critère par ligne. Vous pouvez ajouter, modifier ou supprimer."
            )
        
        else:  # Entrer ses propres critères
            st.info("""
            **Instructions:**
            - Entrez un critère par ligne
            - Soyez spécifique et mesurable
            - Minimum 3 critères, maximum 10 critères
            - Exemples: "Coût d'investissement", "Impact environnemental", "Acceptation sociale"
            """)
            
            criteria_input = st.text_area(
                "Vos critères d'évaluation (un par ligne):",
                height=200,
                placeholder="Exemple:\nCoût d'investissement\nDurée de vie\nImpact environnemental\nAcceptation sociale\nFacilité d'installation",
                help="Un critère par ligne"
            )
        
        # Traitement des critères
        criteria_list = [c.strip() for c in criteria_input.split('\n') if c.strip()]
        
        if len(criteria_list) < 3:
            st.error("⚠️ Veuillez entrer au moins 3 critères")
            st.stop()
        
        if len(criteria_list) > 10:
            st.warning("⚠️ Pour une analyse optimale, limitez à 10 critères maximum")
            criteria_list = criteria_list[:10]
        
        solver.add_criteria(criteria_list)
        
        # Affichage des critères
        st.markdown("---")
        st.markdown("### 📋 Liste de vos critères")
        
        df_criteria = pd.DataFrame({
            'Critère': criteria_list,
            'N°': range(1, len(criteria_list) + 1)
        })
        
        st.dataframe(df_criteria, use_container_width=True, hide_index=True)
        
        # Sélection Best/Worst
        st.markdown("---")
        st.markdown("### 🎯 Étape 2: Sélectionnez le Meilleur et le Pire critère")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Le MEILLEUR critère**")
            st.markdown("*Le plus important pour votre décision*")
            best_crit = st.selectbox(
                "Sélectionnez:",
                criteria_list,
                index=0,
                key="best_select",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**Le PIRE critère**")
            st.markdown("*Le moins important pour votre décision*")
            worst_crit = st.selectbox(
                "Sélectionnez:",
                criteria_list,
                index=len(criteria_list)-1,
                key="worst_select",
                label_visibility="collapsed"
            )
        
        # Validation
        if best_crit == worst_crit:
            st.error("❌ Le meilleur et le pire critère doivent être différents!")
        else:
            solver.set_best_worst(best_crit, worst_crit)
            st.success(f"✅ Sélection enregistrée: **{best_crit}** (Meilleur) vs **{worst_crit}** (Pire)")
        
        # Exemple d'échelle
        with st.expander("📚 Exemple de réflexion pour Best/Worst"):
            st.markdown("""
            **Pour l'énergie solaire au Maroc:**
            - **Meilleur critère:** Rayonnement solaire (sans soleil, pas d'énergie!)
            - **Pire critère:** Distance aux routes (peut être compensée)
            
            **Pour un projet industriel:**
            - **Meilleur critère:** Rentabilité économique
            - **Pire critère:** Complexité administrative
            
            **Pensez à:** Quel critère est INDISPENSABLE? Quel critère est NÉGLIGEABLE?
            """)
    
    # ============================================
    # TAB 2: COMPARAISONS FLOUES
    # ============================================
    with tab2:
        st.markdown("### ⚖️ Étape 3: Matrices de Comparaison Floues")
        
        if not solver.criteria:
            st.warning("⚠️ Veuillez d'abord définir les critères dans l'onglet 1.")
            st.stop()
        
        st.info(f"""
        **Instructions:**
        1. Comparez chaque critère avec **{solver.best_criterion}** (colonne de gauche)
        2. Comparez chaque critère avec **{solver.worst_criterion}** (colonne de droite)
        3. Utilisez l'échelle floue dans la sidebar
        4. Pensez: "Combien de fois le Best est-il plus important que ce critère?"
        """)
        
        # Deux colonnes pour les deux matrices
        col1, col2 = st.columns(2)
        
        # Matrice 1: Best-to-Others
        with col1:
            st.markdown(f"#### 🎯 {solver.best_criterion} vs Autres")
            st.markdown("*Combien le MEILLEUR est-il plus important?*")
            
            BO_comparisons = []
            
            for i, criterion in enumerate(solver.criteria):
                if criterion == solver.best_criterion:
                    BO_comparisons.append(solver.fuzzy_scales['1'])
                    continue
                
                # Widget de sélection avec explication - CORRIGÉ
                comparison = st.selectbox(
                    f"{criterion}:",
                    options=list(solver.fuzzy_scales.keys()),
                    format_func=lambda x: f"{x} - {solver.scale_labels[x]}",
                    key=f"BO_{criterion}_{i}",
                    index=2  # Par défaut à "Modéré"
                )
                
                fuzzy_val = solver.fuzzy_scales[comparison]
                BO_comparisons.append(fuzzy_val)
                
                # Affichage du nombre flou
                st.caption(f"Nombre flou: {fuzzy_val}")
        
        # Matrice 2: Others-to-Worst
        with col2:
            st.markdown(f"#### ⚠️ Autres vs {solver.worst_criterion}")
            st.markdown("*Combien ce critère est-il plus important que le PIRE?*")
            
            OW_comparisons = []
            
            for i, criterion in enumerate(solver.criteria):
                if criterion == solver.worst_criterion:
                    OW_comparisons.append(solver.fuzzy_scales['1'])
                    continue
                
                # Widget de sélection - CORRIGÉ
                comparison = st.selectbox(
                    f"{criterion}:",
                    options=list(solver.fuzzy_scales.keys()),
                    format_func=lambda x: f"{x} - {solver.scale_labels[x]}",
                    key=f"OW_{criterion}_{i}",
                    index=2
                )
                
                fuzzy_val = solver.fuzzy_scales[comparison]
                OW_comparisons.append(fuzzy_val)
                
                st.caption(f"Nombre flou: {fuzzy_val}")
        
        solver.set_comparisons(BO_comparisons, OW_comparisons)
        
        # Bouton de calcul
        st.markdown("---")
        if st.button("🔍 CALCULER LES POIDS ET ANALYSER LA COHÉRENCE", 
                    type="primary", 
                    use_container_width=True):
            
            with st.spinner("Résolution Fuzzy BWM en cours..."):
                weights, consistency, cr = solver.solve(alpha=alpha)
                
                if weights is not None:
                    st.session_state.weights = weights
                    st.session_state.consistency = consistency
                    st.session_state.consistency_ratio = cr
                    
                    st.success("✅ Calcul terminé avec succès!")
                    st.balloons()
                else:
                    st.error("❌ Échec du calcul. Vérifiez vos comparaisons.")
    
    # ============================================
    # TAB 3: RÉSULTATS ET COHÉRENCE DÉTAILLÉE
    # ============================================
    with tab3:
        st.markdown("### 📊 Étape 4: Résultats et Analyse de Cohérence")
        
        if 'weights' not in st.session_state:
            st.warning("⚠️ Veuillez d'abord calculer les poids dans l'onglet 2.")
        else:
            weights = st.session_state.weights
            consistency = st.session_state.consistency
            cr = st.session_state.consistency_ratio
            
            # Section 1: Indicateurs de cohérence DÉTAILLÉS
            st.markdown("---")
            st.markdown("#### 📈 Analyse de la Cohérence")
            
            # Cartes d'indicateurs
            col1, col2, col3 = st.columns(3)
            
            with col1:
                xi_class = "good-consistency" if consistency < 0.2 else "bad-consistency"
                st.markdown(f"""
                <div class="consistency-card {xi_class}">
                    <h4>Indice de Cohérence ξ</h4>
                    <h2>{consistency:.4f}</h2>
                    <p>{"✅ Acceptable" if consistency < 0.2 else "❌ À revoir"}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                cr_class = "good-consistency" if cr < 0.2 else "bad-consistency"
                st.markdown(f"""
                <div class="consistency-card {cr_class}">
                    <h4>Ratio de Cohérence CR</h4>
                    <h2>{cr:.3f}</h2>
                    <p>{"✅ Acceptable" if cr < 0.2 else "❌ Inacceptable" if cr >= 0.2 else "⚠️ Limite"}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                status = "✅ BONNE" if cr < 0.1 else "👍 ACCEPTABLE" if cr < 0.2 else "❌ MAUVAISE"
                color = "green" if cr < 0.1 else "orange" if cr < 0.2 else "red"
                st.markdown(f"""
                <div class="consistency-card">
                    <h4>Qualité Globale</h4>
                    <h2 style="color:{color}">{status}</h2>
                    <p>de la cohérence</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Interprétation DÉTAILLÉE de la cohérence
            st.markdown("---")
            st.markdown("#### 📚 Interprétation Théorique et Pratique")
            
            interpretation = solver.get_consistency_interpretation(consistency, cr)
            st.markdown(interpretation, unsafe_allow_html=True)
            
            # Détection des incohérences spécifiques
            inconsistencies = solver.check_specific_inconsistencies()
            
            if inconsistencies:
                st.markdown("---")
                st.markdown("#### 🔍 Incohérences Détectées")
                
                for inc in inconsistencies:
                    with st.expander(f"Problème avec: {inc['critere']}"):
                        st.markdown(f"""
                        **Comparaisons entrées:**
                        - ã_Bj (Best vs {inc['critere']}) = {inc['a_Bj']}
                        - ã_jW ({inc['critere']} vs Worst) = {inc['a_jW']}
                        - ã_BW (Best vs Worst) = {inc['a_BW']}
                        
                        **Problème:** {inc['probleme']}
                        
                        **Vérification:** {inc['a_Bj']} × {inc['a_jW']} ≈ {inc['produit_approx']}
                        
                        **Devrait être proche de:** {inc['a_BW']}
                        
                        **Suggestion:** Ajustez ã_Bj ou ã_jW pour que leur produit soit proche de ã_BW
                        """)
            
            # Section 2: Résultats des poids
            st.markdown("---")
            st.markdown("#### 🏆 Poids des Critères")
            
            # Table des résultats
            results_df = pd.DataFrame({
                'Critère': solver.criteria,
                'Poids': weights,
                'Pourcentage (%)': weights * 100,
                'Rang': np.argsort(-weights) + 1
            }).sort_values('Poids', ascending=False)
            
            # Formatage de la table
            st.dataframe(
                results_df.style.format({
                    'Poids': '{:.4f}',
                    'Pourcentage (%)': '{:.2f}%'
                }).bar(subset=['Poids'], color='#5fba7d'),
                use_container_width=True,
                height=400
            )
            
            # Section 3: Visualisations
            st.markdown("---")
            st.markdown("#### 📊 Visualisations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Diagramme en barres
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=results_df['Critère'],
                        y=results_df['Pourcentage (%)'],
                        marker_color=['#1e3c72' if crit == solver.best_criterion 
                                     else '#dc3545' if crit == solver.worst_criterion 
                                     else '#5fba7d' for crit in results_df['Critère']],
                        text=results_df['Pourcentage (%)'].round(1).astype(str) + '%',
                        textposition='auto',
                    )
                ])
                
                fig_bar.update_layout(
                    title="Distribution des poids (%)",
                    xaxis_title="Critères",
                    yaxis_title="Poids (%)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Diagramme radar
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=weights * 100,
                    theta=solver.criteria,
                    fill='toself',
                    line_color='#5fba7d',
                    fillcolor='rgba(95, 186, 125, 0.4)'
                ))
                
                fig_radar.update_layout(
                    title="Profil des poids - Diagramme radar",
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, max(weights)*120]
                        )
                    ),
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Section 4: Recommandations basées sur les résultats
            st.markdown("---")
            st.markdown("#### 💡 Recommandations Stratégiques")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Priorités d'action:**")
                top_3 = results_df.head(3)
                for idx, row in top_3.iterrows():
                    st.markdown(f"1. **{row['Critère']}** ({row['Pourcentage (%)']:.1f}%)")
                    st.markdown(f"   *Focus principal pour maximiser l'impact*")
            
            with col2:
                st.markdown("**Points à considérer:**")
                bottom_3 = results_df.tail(3)
                for idx, row in bottom_3.iterrows():
                    st.markdown(f"- {row['Critère']} ({row['Pourcentage (%)']:.1f}%)")
                    st.markdown(f"  *Impact limité sur la décision globale*")
    
    # ============================================
    # TAB 4: EXPORT DES RÉSULTATS
    # ============================================
    with tab4:
        st.markdown("### 💾 Étape 5: Export et Partage des Résultats")
        
        if 'weights' not in st.session_state:
            st.warning("Aucun résultat à exporter. Calculez d'abord les poids.")
        else:
            weights = st.session_state.weights
            consistency = st.session_state.consistency
            cr = st.session_state.consistency_ratio
            
            # Format d'export
            export_format = st.radio(
                "Choisissez le format d'export:",
                ["📊 CSV (Excel)", "📝 Rapport HTML", "🔤 JSON (Technique)", "📋 Résumé texte"],
                horizontal=True
            )
            
            if export_format == "📊 CSV (Excel)":
                results_df = pd.DataFrame({
                    'Critère': solver.criteria,
                    'Poids': weights,
                    'Pourcentage_%': weights * 100,
                    'Rang': np.argsort(-weights) + 1,
                    'Best': [1 if crit == solver.best_criterion else 0 for crit in solver.criteria],
                    'Worst': [1 if crit == solver.worst_criterion else 0 for crit in solver.criteria]
                }).sort_values('Poids', ascending=False)
                
                csv = results_df.to_csv(index=False, encoding='utf-8-sig')
                
                st.download_button(
                    label="📥 Télécharger CSV",
                    data=csv,
                    file_name="fuzzy_bwm_resultats_complets.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Aperçu
                st.dataframe(results_df.head(), use_container_width=True)
            
            elif export_format == "📝 Rapport HTML":
                # Génération d'un rapport HTML complet
                current_date = datetime.datetime.now().strftime('%d/%m/%Y %H:%M')
                
                report_html = f"""
                <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        .header {{ text-align: center; border-bottom: 3px solid #1e3c72; padding-bottom: 20px; }}
                        .section {{ margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }}
                        .criteria-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                        .criteria-table th, .criteria-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                        .criteria-table th {{ background-color: #1e3c72; color: white; }}
                        .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; }}
                        .good {{ color: green; font-weight: bold; }}
                        .warning {{ color: orange; font-weight: bold; }}
                        .bad {{ color: red; font-weight: bold; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>Rapport d'Analyse Fuzzy BWM</h1>
                        <h3>Production d'Hydrogène Vert - Maroc</h3>
                        <p>Date: {current_date}</p>
                    </div>
                    
                    <div class="section">
                        <h2>📋 Résumé Exécutif</h2>
                        <p><strong>Nombre de critères:</strong> {len(solver.criteria)}</p>
                        <p><strong>Meilleur critère:</strong> {solver.best_criterion}</p>
                        <p><strong>Pire critère:</strong> {solver.worst_criterion}</p>
                        <p><strong>Indice de cohérence (ξ):</strong> <span class="{'good' if consistency<0.2 else 'warning' if consistency<0.3 else 'bad'}">{consistency:.4f}</span></p>
                        <p><strong>Ratio de cohérence (CR):</strong> <span class="{'good' if cr<0.1 else 'warning' if cr<0.2 else 'bad'}">{cr:.3f}</span></p>
                        <p><strong>Qualité de cohérence:</strong> {'Très bonne' if cr<0.1 else 'Acceptable' if cr<0.2 else 'À revoir'}</p>
                    </div>
                    
                    <div class="section">
                        <h2>📊 Résultats Détaillés</h2>
                        <table class="criteria-table">
                            <tr>
                                <th>Critère</th>
                                <th>Poids</th>
                                <th>%</th>
                                <th>Rang</th>
                            </tr>
                """
                
                sorted_crit_weights = sorted(zip(solver.criteria, weights), key=lambda x: x[1], reverse=True)
                for i, (crit, w) in enumerate(sorted_crit_weights):
                    is_best = "🏆" if crit == solver.best_criterion else ""
                    is_worst = "⚠️" if crit == solver.worst_criterion else ""
                    report_html += f"""
                            <tr>
                                <td>{is_best}{is_worst} {crit}</td>
                                <td>{w:.4f}</td>
                                <td>{w*100:.2f}%</td>
                                <td>{i+1}</td>
                            </tr>
                    """
                
                report_html += f"""
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>💡 Recommandations</h2>
                        <div class="highlight">
                            <h3>Priorités:</h3>
                            <ol>
                """
                
                sorted_indices = np.argsort(-weights)
                for i in range(min(3, len(weights))):
                    idx = sorted_indices[i]
                    report_html += f"""
                                <li><strong>{solver.criteria[idx]}</strong> ({weights[idx]*100:.1f}%) - Focus principal</li>
                    """
                
                report_html += """
                            </ol>
                            <h3>Points à surveiller:</h3>
                            <ul>
                """
                
                if cr >= 0.2:
                    report_html += """
                                <li><strong>Cohérence insuffisante:</strong> Revoir les comparaisons pour améliorer la fiabilité</li>
                    """
                
                report_html += f"""
                            </ul>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>📈 Méthodologie</h2>
                        <p><strong>Méthode utilisée:</strong> Fuzzy Best-Worst Method (Fuzzy BWM)</p>
                        <p><strong>Niveau α:</strong> {alpha}</p>
                        <p><strong>Date de calcul:</strong> {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                        <p><em>Rapport généré automatiquement par l'application Fuzzy BWM pour la production d'H₂ vert au Maroc</em></p>
                    </div>
                </body>
                </html>
                """
                
                st.download_button(
                    label="📥 Télécharger Rapport HTML",
                    data=report_html,
                    file_name="rapport_fuzzy_bwm.html",
                    mime="text/html",
                    use_container_width=True
                )
                
                # Aperçu du rapport
                with st.expander("Aperçu du rapport"):
                    from streamlit.components.v1 import html
                    html(report_html, height=600, scrolling=True)
            
            elif export_format == "🔤 JSON (Technique)":
                import json
                
                export_data = {
                    "metadata": {
                        "method": "Fuzzy_BWM",
                        "version": "1.0",
                        "date": datetime.datetime.now().isoformat(),
                        "alpha": alpha,
                        "best_criterion": solver.best_criterion,
                        "worst_criterion": solver.worst_criterion
                    },
                    "consistency": {
                        "xi": float(consistency),
                        "cr": float(cr),
                        "interpretation": solver.get_consistency_interpretation(consistency, cr).split('\n')[0]
                    },
                    "criteria": solver.criteria,
                    "weights": weights.tolist(),
                    "comparisons": {
                        "BO": [str(f) for f in solver.BO_matrix],
                        "OW": [str(f) for f in solver.OW_matrix]
                    }
                }
                
                json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="📥 Télécharger JSON",
                    data=json_str,
                    file_name="fuzzy_bwm_data.json",
                    mime="application/json",
                    use_container_width=True
                )
                
                st.code(json_str[:500] + "...", language="json")
            
            else:  # Résumé texte
                summary = f"""
                ============================================
                RAPPORT FUZZY BWM - PRODUCTION H₂ VERT MAROC
                ============================================
                
                DATE: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}
                
                📊 RÉSUMÉ
                ---------
                • Nombre de critères: {len(solver.criteria)}
                • Meilleur critère: {solver.best_criterion}
                • Pire critère: {solver.worst_criterion}
                • Indice de cohérence (ξ): {consistency:.4f}
                • Ratio de cohérence (CR): {cr:.3f}
                • Qualité: {'Très bonne' if cr<0.1 else 'Acceptable' if cr<0.2 else 'À améliorer'}
                
                🏆 CLASSEMENT DES CRITÈRES
                -------------------------
                """
                
                sorted_indices = np.argsort(-weights)
                for i, idx in enumerate(sorted_indices):
                    rank = i + 1
                    crit = solver.criteria[idx]
                    weight = weights[idx]
                    summary += f"{rank}. {crit}: {weight:.4f} ({weight*100:.1f}%)\n"
                
                summary += f"""
                
                💡 RECOMMANDATIONS
                ------------------
                • Priorité absolue: {solver.criteria[sorted_indices[0]]}
                • Secondaire: {solver.criteria[sorted_indices[1]]}
                • Tertiaire: {solver.criteria[sorted_indices[2]]}
                
                {"• ATTENTION: Cohérence faible - Revoir les comparaisons" if cr >= 0.2 else "• Cohérence satisfaisante"}
                
                ============================================
                Fin du rapport
                """
                
                st.download_button(
                    label="📥 Télécharger Résumé Texte",
                    data=summary,
                    file_name="resume_fuzzy_bwm.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.text_area("Aperçu du résumé:", summary, height=300)

# ============================================
# EXÉCUTION
# ============================================

if __name__ == "__main__":
    main()
