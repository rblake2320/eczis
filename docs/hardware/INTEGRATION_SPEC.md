# UNFILED-IP
# Physical Rig Integration Contract

A physical adapter must implement `CaptureRig` and provide:

- linearized RAW frames with black level, saturation, dead-pixel, and invalid masks;
- calibrated subpixel XY positions with repeatability and backlash metadata;
- focus position and repeatability;
- LED direction, intensity, wavelength, and timing;
- exposure, gain, timestamp, and trigger metadata;
- dark-frame, flat-field, PSF, noise, nonlinearity, and shift-lattice calibration; and
- immutable capture identifiers.

Before any real-world claim, rerun operator mismatch, registration, reconstruction, gate, provenance, tamper, and microscope-ground-truth tests on physical captures. Synthetic-only claims must remain visibly separated from physical results.
