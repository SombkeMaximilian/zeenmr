use crate::alignment::assignment::Assignment;
use crate::alignment::feature::FeatureMap;
use crate::alignment::solving::{FeatureVariable, VariableMap};
use good_lp::{
    Constraint, Expression, ProblemVariables, Solution, SolverModel, constraint, default_solver,
    variable,
};
use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct AlignmentProblem {
    problem: ProblemVariables,
    variables: Vec<VariableMap>,
    objective: Expression,
}

impl AlignmentProblem {
    pub(crate) fn add_assignments(&mut self, feature_map: FeatureMap) {
        let feature_variables = feature_map
            .assignments()
            .iter()
            .map(|assignment| {
                FeatureVariable::new(
                    assignment.feature_a(),
                    assignment.feature_b(),
                    assignment.similarity(),
                    self.problem.add(variable().binary()),
                )
            })
            .collect::<Vec<_>>();
        let variable_map = VariableMap::new(
            feature_map.layer_i(),
            feature_map.layer_j(),
            feature_variables,
        );
        self.variables.push(variable_map);
        self.objective = self
            .variables
            .iter()
            .map(|layer_variable_map| {
                layer_variable_map
                    .variables()
                    .iter()
                    .map(|feature_variable| feature_variable.expression())
                    .sum::<Expression>()
            })
            .sum();
    }

    pub(crate) fn best_assignment(self) -> Vec<FeatureMap> {
        let constraints = self.constraints();
        let objective = self.objective;
        let solution = self
            .problem
            .maximise(objective)
            .using(default_solver)
            .with_all(constraints)
            .solve()
            .unwrap();

        self.variables
            .iter()
            .map(|layer_variable_map| {
                let selected_assignments = layer_variable_map
                    .variables()
                    .iter()
                    .filter_map(|feature_variable| {
                        if solution.value(feature_variable.variable()) > 0.5 {
                            Some(Assignment::new(
                                feature_variable.feature_a(),
                                feature_variable.feature_b(),
                                feature_variable.similarity(),
                            ))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                FeatureMap::new(
                    layer_variable_map.layer_i(),
                    layer_variable_map.layer_j(),
                    selected_assignments,
                )
            })
            .collect::<Vec<_>>()
    }

    fn constraints(&self) -> Vec<Constraint> {
        let mut constraints = Vec::new();
        self.variables.iter().for_each(|variable_map| {
            let mut feature_a_constraints = HashMap::new();
            let mut feature_b_constraints = HashMap::new();
            variable_map
                .variables()
                .iter()
                .for_each(|feature_variable| {
                    feature_a_constraints
                        .entry(feature_variable.feature_a())
                        .or_insert_with(Vec::new)
                        .push(feature_variable.variable());
                    feature_b_constraints
                        .entry(feature_variable.feature_b())
                        .or_insert_with(Vec::new)
                        .push(feature_variable.variable());
                });
            feature_a_constraints
                .iter()
                .for_each(|(_, variables)| {
                    constraints.push(constraint!(variables.iter().sum::<Expression>() <= 1));
                });
            feature_b_constraints
                .iter()
                .for_each(|(_, variables)| {
                    constraints.push(constraint!(variables.iter().sum::<Expression>() <= 1));
                });
        });

        constraints
    }
}
