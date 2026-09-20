use std::{
    num::NonZeroU32,
    ops::{Deref, DerefMut, Sub},
    rc::Rc,
    time::Duration,
};

use crate::spring::DEFAULT_SPRING_EPSILON;
use crate::{SpringAnimation, SpringConfig, SpringState, SpringTarget};

/// Creates a duration from a number of whole seconds.
pub const fn secs(seconds: u64) -> Duration {
    Duration::from_secs(seconds)
}

/// Creates a duration from a number of whole milliseconds.
pub const fn millis(milliseconds: u64) -> Duration {
    Duration::from_millis(milliseconds)
}

/// Animation progress is normalized before easing and may overshoot afterward.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Progress(f32);

impl Progress {
    /// The beginning of an animation.
    pub const START: Self = Self(0.0);

    /// The end of an animation.
    pub const END: Self = Self(1.0);

    /// Returns progress clamped to the normalized range.
    pub fn clamped(value: f32) -> Self {
        assert!(!value.is_nan(), "progress must not be NaN");
        Self(value.clamp(Self::START.0, Self::END.0))
    }

    /// Returns the underlying progress value.
    pub const fn get(self) -> f32 {
        self.0
    }

    /// Returns whether the value is at least one; use [`MotionSample::is_active`]
    /// to check if motion has finished.
    pub const fn is_complete(self) -> bool {
        self.0 >= Self::END.0
    }

    fn eased(value: f32) -> Self {
        assert!(value.is_finite(), "easing must return a finite value");
        Self(value)
    }
}

/// Creates duration-based motion with easing or a sampled spring.
pub trait MotionDurationExt {
    /// Creates motion with this duration and the supplied easing function.
    fn with_easing(self, easing: impl Fn(f32) -> f32 + 'static) -> Motion;

    /// Samples a spring over this duration, ending when the duration expires.
    fn with_spring(self, config: SpringConfig) -> Motion;
}

impl MotionDurationExt for Duration {
    fn with_easing(self, easing: impl Fn(f32) -> f32 + 'static) -> Motion {
        Motion::new(self).with_easing(easing)
    }

    fn with_spring(self, config: SpringConfig) -> Motion {
        Motion::new(self).with_spring(config)
    }
}

/// The former name of [`MotionDurationExt`].
pub use MotionDurationExt as DurationWithEasing;

/// Maps linear progress to eased progress.
#[derive(Clone)]
pub struct Easing(Rc<dyn Fn(f32) -> f32>);

impl Easing {
    /// Creates an easing function.
    pub fn new(easing: impl Fn(f32) -> f32 + 'static) -> Self {
        Self(Rc::new(easing))
    }

    /// Evaluates normalized progress without clamping the eased result.
    pub fn sample(&self, progress: Progress) -> Progress {
        Progress::eased((self.0)(progress.get()))
    }
}

impl Default for Easing {
    fn default() -> Self {
        Self::new(crate::linear)
    }
}

/// The duration and easing curve for one directional motion pass.
///
/// Local pass time advances from zero to one. The easing function maps that
/// local time to normalized presentation progress and may be non-monotonic.
/// When this is the reverse pass of alternating motion, [`Motion`] maps the
/// eased result back toward the origin.
#[derive(Clone)]
pub struct MotionPass {
    duration: Duration,
    easing: Easing,
}

impl MotionPass {
    /// Creates a linear pass with the supplied duration.
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            easing: Easing::default(),
        }
    }

    /// Configures this pass's easing function.
    pub fn with_easing(mut self, easing: impl Fn(f32) -> f32 + 'static) -> Self {
        self.easing = Easing::new(easing);
        self
    }

    /// Returns this pass's duration.
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Returns this pass's easing function.
    pub fn easing(&self) -> &Easing {
        &self.easing
    }

    fn duration_nanos(&self) -> u128 {
        self.duration.as_nanos()
    }

    fn sample(&self, elapsed_nanos: u128) -> Progress {
        let duration_nanos = self.duration_nanos();
        debug_assert!(duration_nanos > 0);
        debug_assert!(elapsed_nanos < duration_nanos);

        let linear = elapsed_nanos as f64 / duration_nanos as f64;
        self.easing.sample(Progress::clamped(linear as f32))
    }
}

impl From<Duration> for MotionPass {
    fn from(duration: Duration) -> Self {
        Self::new(duration)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IterationCount {
    Finite(NonZeroU32),
    Forever,
}

impl Default for IterationCount {
    fn default() -> Self {
        Self::Finite(NonZeroU32::MIN)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum IterationMode {
    #[default]
    Restart,
    Alternate,
}

/// The result of evaluating motion at a point in time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotionSample {
    /// Normalized presentation progress after easing.
    pub progress: Progress,

    /// Whether another sample may produce a different value.
    pub is_active: bool,
}

/// Configuration for duration-based playback.
#[derive(Clone)]
pub struct DurationDescription {
    forward: MotionPass,
    reverse: Option<MotionPass>,
    delay: Duration,
    iterations: IterationCount,
    iteration_mode: IterationMode,
}

/// Configuration for motion driven by a settling spring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpringDescription {
    /// The spring's physical parameters.
    pub(crate) config: SpringConfig,

    /// The distance and velocity threshold for settling.
    pub(crate) epsilon: f32,

    settle_after: Duration,
}

/// A motion with methods determined by its description type.
#[derive(Clone, Debug)]
pub struct Motion<Description = DurationDescription> {
    description: Description,
}

impl<Description> Deref for Motion<Description> {
    type Target = Description;

    fn deref(&self) -> &Self::Target {
        &self.description
    }
}

impl<Description> DerefMut for Motion<Description> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.description
    }
}

impl Motion<DurationDescription> {
    /// Creates one linear motion pass with the supplied duration.
    pub fn new(duration: Duration) -> Self {
        Self {
            description: DurationDescription {
                forward: MotionPass::new(duration),
                reverse: None,
                delay: Duration::ZERO,
                iterations: IterationCount::default(),
                iteration_mode: IterationMode::default(),
            },
        }
    }

    /// Configures the forward pass's easing function.
    ///
    /// An alternating motion without an explicit reverse pass also uses this
    /// easing for its backward iterations.
    pub fn with_easing(mut self, easing: impl Fn(f32) -> f32 + 'static) -> Self {
        self.description.forward.easing = Easing::new(easing);
        self
    }

    /// Replaces easing with a spring sampled over this motion's duration.
    /// Use [`Motion::spring`] for a spring that settles and preserves velocity across retargets.
    pub fn with_spring(self, config: SpringConfig) -> Self {
        let duration = self.forward.duration.as_secs_f32();
        let initial_state = SpringState {
            position: 0.0,
            velocity: 0.0,
        };

        self.with_easing(move |progress| {
            if progress <= 0.0 {
                0.0
            } else if progress >= 1.0 {
                1.0
            } else {
                config
                    .step(initial_state, 1.0, progress * duration)
                    .position
            }
        })
    }

    /// Replaces the forward pass.
    pub fn with_forward_pass(mut self, pass: MotionPass) -> Self {
        self.forward = pass;
        self
    }

    /// Configures the pass used for backward legs of alternating playback.
    ///
    /// This does not infer direction from changes to an [`Animated`](crate::Animated)
    /// target. A newly retargeted run begins with the forward pass. Without an
    /// explicit reverse pass, backward legs reuse the forward pass's duration
    /// and easing.
    ///
    /// Reverse local time advances from zero to one and is eased normally. Its
    /// normalized presentation is then mapped as `1 - eased`, so reverse
    /// ease-in begins slowly and accelerates toward the origin.
    pub fn with_reverse_pass(mut self, pass: MotionPass) -> Self {
        self.reverse = Some(pass);
        self
    }

    /// Delays the start of playback once per run.
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    /// Sets the total number of iterations, including the first.
    ///
    /// The default is one. Restarting motion restarts forward-pass local timing
    /// on every iteration. In alternating motion, odd-numbered iterations are
    /// forward and even-numbered iterations are backward. With conventional
    /// easing endpoints, counts of one and three rest at the target while counts
    /// of two and four rest at the origin.
    ///
    /// # Panics
    ///
    /// Panics if `iterations` is zero.
    pub fn iterations(mut self, iterations: u32) -> Self {
        self.iterations = IterationCount::Finite(
            NonZeroU32::new(iterations).expect("motion iterations must be at least 1"),
        );
        self
    }

    /// Repeats this motion indefinitely.
    ///
    /// A motion whose applicable passes all have zero duration becomes inactive
    /// at a deterministic resting presentation.
    pub fn repeat_forever(mut self) -> Self {
        self.iterations = IterationCount::Forever;
        self
    }

    /// Alternates forward and backward iterations.
    ///
    /// Odd-numbered iterations use the forward pass. Even-numbered iterations
    /// use the configured reverse pass, falling back to the forward pass when
    /// none is configured.
    pub fn alternate(mut self) -> Self {
        self.iteration_mode = IterationMode::Alternate;
        self
    }

    /// Returns the forward pass.
    pub fn forward_pass(&self) -> &MotionPass {
        &self.forward
    }

    /// Returns the pass explicitly configured for backward legs of alternating
    /// playback, if any.
    pub fn reverse_pass(&self) -> Option<&MotionPass> {
        self.reverse.as_ref()
    }

    /// Returns the initial delay applied once per run.
    pub fn delay(&self) -> Duration {
        self.delay
    }

    /// Returns the finite total iteration count, or `None` for indefinite motion.
    pub fn iteration_count(&self) -> Option<u32> {
        match self.iterations {
            IterationCount::Finite(iterations) => Some(iterations.get()),
            IterationCount::Forever => None,
        }
    }

    /// Returns whether this motion repeats indefinitely.
    pub fn repeats_forever(&self) -> bool {
        self.iterations == IterationCount::Forever
    }

    /// Returns whether every second iteration runs backward.
    pub fn is_alternating(&self) -> bool {
        self.iteration_mode == IterationMode::Alternate
    }

    fn reverse_pass_or_forward(&self) -> &MotionPass {
        self.reverse.as_ref().unwrap_or(&self.forward)
    }

    fn finite_duration_nanos(&self, iterations: NonZeroU32) -> u128 {
        let iterations = u128::from(iterations.get());
        let forward = self.forward.duration_nanos();

        match self.iteration_mode {
            IterationMode::Restart => forward.saturating_mul(iterations),
            IterationMode::Alternate => {
                let reverse = self.reverse_pass_or_forward().duration_nanos();
                let pairs = iterations / 2;
                let trailing_forward = iterations % 2;
                forward
                    .saturating_add(reverse)
                    .saturating_mul(pairs)
                    .saturating_add(forward.saturating_mul(trailing_forward))
            }
        }
    }

    fn active_pass(&self, elapsed_nanos: u128) -> Option<(&MotionPass, u128, bool)> {
        let forward_duration = self.forward.duration_nanos();

        match self.iteration_mode {
            IterationMode::Restart => (forward_duration > 0)
                .then(|| (&self.forward, elapsed_nanos % forward_duration, false)),
            IterationMode::Alternate => {
                let reverse = self.reverse_pass_or_forward();
                let reverse_duration = reverse.duration_nanos();
                let cycle_duration = forward_duration.saturating_add(reverse_duration);
                if cycle_duration == 0 {
                    return None;
                }

                let elapsed_in_cycle = elapsed_nanos % cycle_duration;
                if elapsed_in_cycle < forward_duration {
                    Some((&self.forward, elapsed_in_cycle, false))
                } else {
                    Some((reverse, elapsed_in_cycle - forward_duration, true))
                }
            }
        }
    }

    fn final_progress(&self) -> Progress {
        match (self.iterations, self.iteration_mode) {
            (IterationCount::Finite(iterations), IterationMode::Alternate)
                if iterations.get().is_multiple_of(2) =>
            {
                Progress::START
            }
            (IterationCount::Finite(_), _) => Progress::END,
            (IterationCount::Forever, _) => Progress::START,
        }
    }

    fn inactive_sample(&self) -> MotionSample {
        MotionSample {
            progress: self.final_progress(),
            is_active: false,
        }
    }

    fn completed_sample(&self) -> MotionSample {
        let progress = match (self.iterations, self.iteration_mode) {
            (IterationCount::Finite(iterations), IterationMode::Alternate)
                if iterations.get().is_multiple_of(2) =>
            {
                let eased = self.reverse_pass_or_forward().easing.sample(Progress::END);
                Progress::clamped(Progress::END.get() - eased.get())
            }
            (IterationCount::Finite(_), _) => self.forward.easing.sample(Progress::END),
            (IterationCount::Forever, _) => Progress::START,
        };

        MotionSample {
            progress,
            is_active: false,
        }
    }

    fn sample_active_pass(
        &self,
        pass: &MotionPass,
        elapsed_nanos: u128,
        reverse: bool,
    ) -> Progress {
        let progress = pass.sample(elapsed_nanos);
        if reverse {
            Progress::clamped(Progress::END.get() - progress.get())
        } else {
            progress
        }
    }

    /// Evaluates this motion after the supplied elapsed time.
    ///
    /// Pass selection, iteration parity, and completion use integer nanosecond
    /// arithmetic. Only the fraction within the selected pass is converted to
    /// floating point for easing.
    pub fn sample(&self, elapsed: Duration) -> MotionSample {
        if elapsed < self.delay {
            return MotionSample {
                progress: Progress::START,
                is_active: true,
            };
        }

        let elapsed_nanos = (elapsed - self.delay).as_nanos();
        if let IterationCount::Finite(iterations) = self.iterations {
            let total_duration = self.finite_duration_nanos(iterations);
            if elapsed_nanos >= total_duration {
                return if total_duration == 0 {
                    self.inactive_sample()
                } else {
                    self.completed_sample()
                };
            }
        }

        let Some((pass, elapsed_in_pass, reverse)) = self.active_pass(elapsed_nanos) else {
            return self.inactive_sample();
        };

        MotionSample {
            progress: self.sample_active_pass(pass, elapsed_in_pass, reverse),
            is_active: true,
        }
    }

    /// Evaluates this motion between two timestamps.
    pub fn sample_at<Time>(&self, started_at: Time, now: Time) -> MotionSample
    where
        Time: Sub<Time, Output = Duration>,
    {
        self.sample(now - started_at)
    }

    pub(crate) fn resting_progress(&self) -> Progress {
        self.final_progress()
    }
}

impl Default for Motion<DurationDescription> {
    fn default() -> Self {
        Self::new(Duration::ZERO)
    }
}

impl From<Duration> for Motion<DurationDescription> {
    fn from(duration: Duration) -> Self {
        Self::new(duration)
    }
}

/// The former name of [`Motion`].
#[deprecated(note = "use Motion")]
pub type MotionInfo = Motion;

impl Motion<SpringDescription> {
    /// Creates a spring motion that runs until it settles.
    pub fn spring(config: SpringConfig) -> Self {
        let epsilon = DEFAULT_SPRING_EPSILON;

        Self {
            description: SpringDescription {
                config,
                epsilon,
                settle_after: config.settle_time(SpringState::default(), 1.0, epsilon),
            },
        }
    }

    /// Returns this spring's physical parameters.
    pub fn config(&self) -> SpringConfig {
        self.config
    }

    /// Returns this spring's settling tolerance.
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Sets the spring's settling tolerance.
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.description.epsilon = epsilon;
        self.description.settle_after =
            self.config
                .settle_time(SpringState::default(), 1.0, epsilon);
        self
    }

    /// Evaluates this spring after the supplied elapsed time, starting from rest at zero progress.
    pub fn sample(&self, elapsed: Duration) -> MotionSample {
        if elapsed >= self.settle_after {
            return MotionSample {
                progress: Progress::END,
                is_active: false,
            };
        }

        let state = self
            .config
            .step(SpringState::default(), 1.0, elapsed.as_secs_f32());

        MotionSample {
            progress: Progress::eased(state.position),
            is_active: true,
        }
    }

    /// Targets a value or projected path with this spring.
    pub fn to<T: SpringTarget>(self, target: T) -> SpringAnimation<T> {
        SpringAnimation {
            motion: self,
            target,
            initial: None,
            playback: crate::SpringPlayback::Running,
        }
    }
}

impl From<SpringConfig> for Motion<SpringDescription> {
    fn from(config: SpringConfig) -> Self {
        Self::spring(config)
    }
}

/// A duration or spring motion that can be sampled through one interface.
/// Each spring sample starts from rest, so retargeting an animated value resets its velocity.
#[derive(Clone)]
pub enum AnyMotion {
    /// Motion that runs for a fixed duration.
    Duration(Motion<DurationDescription>),

    /// Motion that runs until its spring settles.
    Spring(Motion<SpringDescription>),
}

impl AnyMotion {
    /// Evaluates this motion after the supplied elapsed time.
    pub fn sample(&self, elapsed: Duration) -> MotionSample {
        match self {
            Self::Duration(motion) => motion.sample(elapsed),
            Self::Spring(motion) => motion.sample(elapsed),
        }
    }

    /// Evaluates this motion between two timestamps.
    pub fn sample_at<Time>(&self, started_at: Time, now: Time) -> MotionSample
    where
        Time: Sub<Time, Output = Duration>,
    {
        self.sample(now - started_at)
    }

    pub(crate) fn resting_progress(&self) -> Progress {
        match self {
            Self::Duration(motion) => motion.resting_progress(),
            Self::Spring(_) => Progress::END,
        }
    }
}

impl From<Motion<DurationDescription>> for AnyMotion {
    fn from(motion: Motion<DurationDescription>) -> Self {
        Self::Duration(motion)
    }
}

impl From<Motion<SpringDescription>> for AnyMotion {
    fn from(motion: Motion<SpringDescription>) -> Self {
        Self::Spring(motion)
    }
}

impl From<Duration> for AnyMotion {
    fn from(duration: Duration) -> Self {
        Self::Duration(duration.into())
    }
}

impl From<SpringConfig> for AnyMotion {
    fn from(config: SpringConfig) -> Self {
        Self::Spring(config.into())
    }
}

impl From<&Motion<DurationDescription>> for AnyMotion {
    fn from(motion: &Motion<DurationDescription>) -> Self {
        Self::Duration(motion.clone())
    }
}

impl From<&Motion<SpringDescription>> for AnyMotion {
    fn from(motion: &Motion<SpringDescription>) -> Self {
        Self::Spring(motion.clone())
    }
}

impl From<&AnyMotion> for AnyMotion {
    fn from(motion: &AnyMotion) -> Self {
        motion.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(value: f32) -> Progress {
        Progress::clamped(value)
    }

    fn assert_sample(motion: &Motion, elapsed: Duration, value: f32, is_active: bool) {
        assert_eq!(
            motion.sample(elapsed),
            MotionSample {
                progress: progress(value),
                is_active,
            }
        );
    }

    #[test]
    fn creates_durations() {
        assert_eq!(secs(2), Duration::from_secs(2));
        assert_eq!(millis(250), Duration::from_millis(250));
    }

    #[test]
    fn samples_default_one_shot_and_custom_easing() {
        let motion = Duration::from_secs(2).with_easing(|progress| progress * progress);
        assert_eq!(motion.forward_pass().duration(), Duration::from_secs(2));
        assert!(motion.reverse_pass().is_none());
        assert_eq!(motion.delay(), Duration::ZERO);
        assert_eq!(motion.iteration_count(), Some(1));
        assert!(!motion.repeats_forever());
        assert!(!motion.is_alternating());

        assert_sample(&motion, Duration::ZERO, 0.0, true);
        assert_sample(&motion, Duration::from_secs(1), 0.25, true);
        assert_sample(&motion, Duration::from_secs(3), 1.0, false);
        assert_eq!(
            motion.sample_at(Duration::from_secs(3), Duration::from_secs(5)),
            MotionSample {
                progress: Progress::END,
                is_active: false,
            }
        );

        assert_eq!(Progress::clamped(-1.0), Progress::START);
        assert_eq!(Progress::clamped(2.0), Progress::END);
    }

    #[test]
    fn delay_applies_once_at_the_start_of_a_run() {
        let motion = Motion::new(Duration::from_secs(1))
            .with_delay(Duration::from_millis(100))
            .iterations(3);

        assert_sample(&motion, Duration::ZERO, 0.0, true);
        assert_sample(&motion, Duration::from_millis(99), 0.0, true);
        assert_sample(&motion, Duration::from_millis(100), 0.0, true);
        assert_sample(&motion, Duration::from_millis(600), 0.5, true);
        assert_sample(&motion, Duration::from_millis(1_100), 0.0, true);
        assert_sample(&motion, Duration::from_millis(3_100), 1.0, false);
    }

    #[test]
    fn restarting_iterations_have_exact_boundaries_and_stable_completion() {
        let motion = Motion::new(Duration::from_secs(1)).iterations(3);

        for (milliseconds, value, active) in [
            (0, 0.0, true),
            (500, 0.5, true),
            (1_000, 0.0, true),
            (2_000, 0.0, true),
            (2_750, 0.75, true),
            (3_000, 1.0, false),
            (30_000, 1.0, false),
        ] {
            assert_sample(&motion, Duration::from_millis(milliseconds), value, active);
        }
    }

    #[test]
    fn alternating_iterations_reverse_and_settle_by_parity() {
        for (iterations, expected_final) in [(1, 1.0), (2, 0.0), (3, 1.0), (4, 0.0)] {
            let motion = Motion::new(Duration::from_secs(1))
                .iterations(iterations)
                .alternate();
            assert_sample(
                &motion,
                Duration::from_secs(u64::from(iterations)),
                expected_final,
                false,
            );
            assert_sample(&motion, Duration::from_secs(100), expected_final, false);
        }

        let motion = Motion::new(Duration::from_secs(1))
            .iterations(4)
            .alternate();
        for (milliseconds, value) in [
            (0, 0.0),
            (500, 0.5),
            (1_000, 1.0),
            (1_500, 0.5),
            (2_000, 0.0),
            (3_000, 1.0),
        ] {
            assert_sample(&motion, Duration::from_millis(milliseconds), value, true);
        }
    }

    #[test]
    fn asymmetric_alternating_passes_use_local_easing() {
        let motion = Motion::new(Duration::from_secs(1))
            .with_easing(|value| value)
            .with_reverse_pass(
                MotionPass::new(Duration::from_secs(2)).with_easing(|value| value * value),
            )
            .iterations(5)
            .alternate();

        assert_eq!(
            motion.reverse_pass().unwrap().duration(),
            Duration::from_secs(2)
        );
        assert_sample(&motion, Duration::from_millis(500), 0.5, true);
        assert_sample(&motion, Duration::from_secs(1), 1.0, true);
        assert_sample(&motion, Duration::from_secs(2), 0.75, true);
        assert_sample(&motion, Duration::from_secs(3), 0.0, true);
        assert_sample(&motion, Duration::from_millis(6_500), 0.5, true);
        assert_sample(&motion, Duration::from_secs(7), 1.0, false);
    }

    #[test]
    fn infinite_motion_restarts_or_alternates_without_completing() {
        let restarting = Motion::new(Duration::from_secs(1)).repeat_forever();
        assert_sample(&restarting, Duration::from_millis(250), 0.25, true);
        assert_sample(&restarting, Duration::from_secs(1), 0.0, true);
        assert_sample(&restarting, Duration::from_millis(2_500), 0.5, true);

        let alternating = Motion::new(Duration::from_secs(1))
            .repeat_forever()
            .alternate();
        assert_sample(&alternating, Duration::from_millis(1_250), 0.75, true);
        assert_sample(&alternating, Duration::from_secs(2), 0.0, true);
    }

    #[test]
    fn zero_duration_motion_resolves_after_delay_without_spinning() {
        for (iterations, alternate, expected) in [
            (1, false, 1.0),
            (2, false, 1.0),
            (2, true, 0.0),
            (3, true, 1.0),
        ] {
            let mut motion = Motion::new(Duration::ZERO)
                .with_delay(Duration::from_millis(10))
                .iterations(iterations);
            if alternate {
                motion = motion.alternate();
            }
            assert_sample(&motion, Duration::from_millis(9), 0.0, true);
            assert_sample(&motion, Duration::from_millis(10), expected, false);
        }

        let forever = Motion::new(Duration::ZERO).repeat_forever().alternate();
        assert_sample(&forever, Duration::ZERO, 0.0, false);
        assert_sample(&forever, Duration::MAX, 0.0, false);
    }

    #[test]
    fn zero_length_passes_are_skipped_without_iteration() {
        let zero_forward = Motion::new(Duration::ZERO)
            .with_reverse_pass(MotionPass::new(Duration::from_secs(2)))
            .iterations(2)
            .alternate();
        assert_sample(&zero_forward, Duration::ZERO, 1.0, true);
        assert_sample(&zero_forward, Duration::from_secs(1), 0.5, true);
        assert_sample(&zero_forward, Duration::from_secs(2), 0.0, false);

        let zero_reverse = Motion::new(Duration::from_secs(1))
            .with_reverse_pass(MotionPass::new(Duration::ZERO))
            .iterations(3)
            .alternate();
        assert_sample(&zero_reverse, Duration::from_secs(1), 0.0, true);
        assert_sample(&zero_reverse, Duration::from_secs(2), 1.0, false);
    }

    #[test]
    fn nanosecond_and_large_duration_sampling_keep_exact_pass_selection() {
        let nanos = Motion::new(Duration::from_nanos(3))
            .iterations(3)
            .alternate();
        assert_sample(&nanos, Duration::from_nanos(3), 1.0, true);
        assert_sample(&nanos, Duration::from_nanos(6), 0.0, true);
        assert_sample(&nanos, Duration::from_nanos(9), 1.0, false);

        let enormous = Motion::new(Duration::MAX).iterations(u32::MAX);
        assert_sample(&enormous, Duration::MAX, 0.0, true);
    }

    #[test]
    #[should_panic(expected = "motion iterations must be at least 1")]
    fn zero_iterations_are_invalid() {
        let _ = Motion::new(Duration::from_secs(1)).iterations(0);
    }
    #[test]
    fn sampled_and_settling_springs_keep_the_new_motion_api() {
        let config = SpringConfig::new(100.0, 6.0, 1.0);
        let sampled = Motion::new(Duration::from_secs(1)).with_spring(config);
        assert_eq!(sampled.sample(Duration::ZERO).progress, Progress::START);
        assert_eq!(
            sampled.sample(Duration::from_secs(1)).progress,
            Progress::END
        );
        assert!((1..100).any(|step| {
            sampled
                .sample(Duration::from_millis(step * 10))
                .progress
                .get()
                > 1.0
        }));

        let native: AnyMotion = config.into();
        let loose = Motion::spring(config).with_epsilon(0.1);
        let cutoff = loose.settle_after;
        assert!(native.sample(cutoff).is_active);
        assert_eq!(
            AnyMotion::from(loose).sample_at(Duration::ZERO, cutoff),
            MotionSample {
                progress: Progress::END,
                is_active: false
            }
        );
    }
}
