use super::param::ToParams;

pub trait Optimizer {
    fn with(self, optimizable: &mut impl ToParams) -> Self;
    fn step(&mut self, optimizable: &mut impl ToParams);
}
