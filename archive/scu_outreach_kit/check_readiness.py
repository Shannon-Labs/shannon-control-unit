#!/usr/bin/env python3
"""Check readiness for HN post and outreach."""

import yaml
from datetime import datetime

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def check_hn_trigger(trigger):
    """Determine if ready to post on HN."""
    reasons = []
    
    # Check conditions
    compute_ready = trigger.get('compute_secured', False)
    ttt_improvement = trigger.get('time_to_target_improvement')
    overhead = trigger.get('overhead_measured')
    traces = trigger.get('profiler_traces', False)
    
    # Primary condition: compute secured
    if compute_ready:
        return "GO", ["Compute partnership secured"]
    
    # Alternative condition: 7B results ready
    if all([
        ttt_improvement is not None and ttt_improvement >= 10,
        overhead is not None and overhead <= 2,
        traces
    ]):
        return "GO", [
            f"7B results show {ttt_improvement}% improvement",
            f"Overhead at {overhead}% (target ≤2%)",
            "Profiler traces collected"
        ]
    
    # Not ready - explain why
    if ttt_improvement is None:
        reasons.append("❌ 7B time-to-target not measured yet")
    elif ttt_improvement < 10:
        reasons.append(f"❌ Time-to-target only {ttt_improvement}% (need ≥10%)")
    
    if overhead is None:
        reasons.append("❌ Overhead not measured yet")
    elif overhead > 2:
        reasons.append(f"❌ Overhead too high at {overhead}% (need ≤2%)")
    
    if not traces:
        reasons.append("❌ Profiler traces not collected")
    
    if not compute_ready:
        reasons.append("⚠️  No compute partnership secured")
    
    return "NO-GO", reasons

def main():
    config = load_config()
    trigger = config['hn_trigger']
    
    print("SCU HN Readiness Check")
    print("=" * 40)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    status, reasons = check_hn_trigger(trigger)
    
    if status == "GO":
        print("✅ READY TO POST ON HN\n")
        print("Reasons:")
        for reason in reasons:
            print(f"  • {reason}")
        print("\nNext steps:")
        print("1. Finalize HN post text")
        print("2. Choose optimal posting time (Tue-Thu, 9am PT)")
        print("3. Have 3-5 friendlies ready to engage early")
    else:
        print("🛑 NOT READY FOR HN\n")
        print("Blocking issues:")
        for reason in reasons:
            print(f"  {reason}")
        print("\nTo become ready, either:")
        print("  1. Secure compute partnership, OR")
        print("  2. Complete 7B validation with good metrics")
    
    print("\n" + "=" * 40)
    print("Outreach Status:")
    print(f"  • Emails ready: output/emails/")
    print(f"  • Docs ready: output/docs/")
    print(f"  • Plots ready: output/plots/")
    print("\nFocus on direct outreach to hyperscalers first!")

if __name__ == "__main__":
    main()