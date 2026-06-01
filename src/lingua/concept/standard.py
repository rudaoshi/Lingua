

LinguaGraphNodePoses = {
    "AttributePredicator",
    "AppositionalPredicator",
    "FactualPredicator",
    "LogicalPredicator",
    "ReferentialPredicator",
    "ConjunctionalFunctor",
    "ExpressionFunctor",
    "GeneralFunctor",
    "ListFunctor",
    "ModificationalFunctor",
    "DeterminerConstant",
    "InterjectionConstant",
    "ModificationalConstant",
    "NominalConstant",
    "OtherConstant",
    "PunctuationalConstant",
    "SymbolConstant"
}

LinguaGraphAuxNodeLabels = {
    "Apposition",
    "Attribute",
    "Copula",
    "Discourse",
    "Expression",
    "List",
    "Modification",
    "Parataxis",
    "Ref",
    "Vocative",
    "Whether",
}

LinguaGraphEdges = {
    "body",
    "variable",
    "pred.arg.1",
    "pred.arg.2",
    "pred.arg.3",
    "pred.arg.4",
    "func.arg",
    "appositive",
    "attribute",
    "discourse",
    "index",
    "modification",
    "nonsense",
    "punctuation",
    "ref",
    "repeat",
    "vocative",
}

LinguaGraphEdges.update( "as:" + edge for edge in list(LinguaGraphEdges))    
