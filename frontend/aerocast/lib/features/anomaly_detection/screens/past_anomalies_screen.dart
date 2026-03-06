import 'package:flutter/material.dart';

enum AnomalySeverity { low, moderate, high, critical }

class PastAnomalyEvent {
  final String location;
  final String eventName;
  final String pollutant;
  final AnomalySeverity severity;
  final int duration;
  final double maxValue;
  final DateTime start;

  PastAnomalyEvent({
    required this.location,
    required this.eventName,
    required this.pollutant,
    required this.severity,
    required this.duration,
    required this.maxValue,
    required this.start,
  });
}

class PastAnomaliesScreen extends StatefulWidget {
  const PastAnomaliesScreen({super.key});

  @override
  State<PastAnomaliesScreen> createState() => _PastAnomaliesScreenState();
}

class _PastAnomaliesScreenState extends State<PastAnomaliesScreen> {
  String selectedLocation = "BATTARAMULLA";

  final List<PastAnomalyEvent> allEvents = [
    // BATTARAMULLA EVENTS
    PastAnomalyEvent(
      location: "BATTARAMULLA",
      eventName: "Dust Spike (Road Construction)",
      pollutant: "PM10",
      severity: AnomalySeverity.high,
      duration: 3,
      maxValue: 112.5,
      start: DateTime.now().subtract(const Duration(days: 2, hours: 4)),
    ),
    PastAnomalyEvent(
      location: "BATTARAMULLA",
      eventName: "Morning Peak Traffic Anomaly",
      pollutant: "PM2.5",
      severity: AnomalySeverity.critical,
      duration: 4,
      maxValue: 97.0,
      start: DateTime.now().subtract(const Duration(days: 5, hours: 10)),
    ),
    PastAnomalyEvent(
      location: "BATTARAMULLA",
      eventName: "Late Evening Sulfur Spike",
      pollutant: "SO2",
      severity: AnomalySeverity.moderate,
      duration: 2,
      maxValue: 4.2,
      start: DateTime.now().subtract(const Duration(days: 12)),
    ),
    PastAnomalyEvent(
      location: "BATTARAMULLA",
      eventName: "Mid-day Photo-chemical Event",
      pollutant: "O3",
      severity: AnomalySeverity.low,
      duration: 1,
      maxValue: 2.1,
      start: DateTime.now().subtract(const Duration(days: 18)),
    ),

    // KANDY EVENTS
    PastAnomalyEvent(
      location: "KANDY",
      eventName: "Valley Trapped Emissions",
      pollutant: "PM2.5",
      severity: AnomalySeverity.critical,
      duration: 8,
      maxValue: 145.0,
      start: DateTime.now().subtract(const Duration(days: 3, hours: 2)),
    ),
    PastAnomalyEvent(
      location: "KANDY",
      eventName: "Industrial Cluster Anomaly",
      pollutant: "SO2",
      severity: AnomalySeverity.high,
      duration: 5,
      maxValue: 6.8,
      start: DateTime.now().subtract(const Duration(days: 7)),
    ),
    PastAnomalyEvent(
      location: "KANDY",
      eventName: "Roadside Dust Concentration",
      pollutant: "PM10",
      severity: AnomalySeverity.moderate,
      duration: 4,
      maxValue: 88.0,
      start: DateTime.now().subtract(const Duration(days: 14, hours: 5)),
    ),
    PastAnomalyEvent(
      location: "KANDY",
      eventName: "Minor Ozone Threshold Drift",
      pollutant: "O3",
      severity: AnomalySeverity.low,
      duration: 2,
      maxValue: 1.8,
      start: DateTime.now().subtract(const Duration(days: 25)),
    ),
  ];

  Color _getSeverityColor(AnomalySeverity severity) {
    switch (severity) {
      case AnomalySeverity.low: return Colors.yellow.shade700;
      case AnomalySeverity.moderate: return Colors.orange;
      case AnomalySeverity.high: return Colors.red;
      case AnomalySeverity.critical: return const Color(0xFF800000); // Dark Maroon
    }
  }

  @override
  Widget build(BuildContext context) {
    final filteredEvents = allEvents.where((e) => e.location == selectedLocation).toList();

    return Scaffold(
      backgroundColor: const Color(0xFFF8FAFC),
      appBar: AppBar(
        title: const Text("Past Detected Anomalies", style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        elevation: 0,
        foregroundColor: Colors.black,
      ),
      body: Column(
        children: [
          _buildLocationSelector(),
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              itemCount: filteredEvents.length,
              itemBuilder: (context, index) {
                final event = filteredEvents[index];
                return _buildAnomalyCard(event);
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLocationSelector() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
      color: Colors.white,
      child: Row(
        children: [
          _buildSelectButton("BATTARAMULLA"),
          const SizedBox(width: 12),
          _buildSelectButton("KANDY"),
        ],
      ),
    );
  }

  Widget _buildSelectButton(String loc) {
    bool isSelected = selectedLocation == loc;
    return Expanded(
      child: InkWell(
        onTap: () => setState(() => selectedLocation = loc),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: isSelected ? Colors.blue.shade600 : Colors.white,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: isSelected ? Colors.blue.shade600 : Colors.grey.shade300),
            boxShadow: isSelected ? [BoxShadow(color: Colors.blue.withOpacity(0.3), blurRadius: 8, offset: const Offset(0, 4))] : [],
          ),
          child: Text(
            loc,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 13,
              color: isSelected ? Colors.white : Colors.black54,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildAnomalyCard(PastAnomalyEvent event) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 10, offset: const Offset(0, 4))],
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      event.location,
                      style: TextStyle(fontWeight: FontWeight.bold, color: Colors.blue.shade700, letterSpacing: 1.1, fontSize: 13),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      event.eventName,
                      style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Color(0xFF2D3142)),
                    ),
                  ],
                ),
                Column(
                  children: [
                    Icon(Icons.warning_amber_rounded, color: Colors.red.shade400, size: 28),
                    const SizedBox(height: 4),
                    Container(
                      width: 12,
                      height: 12,
                      decoration: BoxDecoration(
                        color: _getSeverityColor(event.severity),
                        shape: BoxShape.circle,
                        boxShadow: [BoxShadow(color: _getSeverityColor(event.severity).withOpacity(0.4), blurRadius: 4)],
                      ),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(child: _buildMiniStat("Pollutant", event.pollutant)),
                Expanded(child: _buildMiniStat("Duration", "${event.duration}h")),
                Expanded(child: _buildMiniStat("Max Value", "${event.maxValue}")),
                Expanded(child: _buildMiniStat("Severity", event.severity.name.toUpperCase(), color: _getSeverityColor(event.severity))),
              ],
            ),
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 16),
              child: Divider(height: 1),
            ),
            Row(
              children: [
                const Icon(Icons.calendar_today_outlined, size: 14, color: Colors.grey),
                const SizedBox(width: 8),
                Text(
                  "Detected: ${event.start.day}/${event.start.month}/${event.start.year} at ${event.start.hour}:00",
                  style: const TextStyle(color: Colors.grey, fontSize: 13),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMiniStat(String label, String value, {Color? color}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: const TextStyle(color: Colors.grey, fontSize: 11, fontWeight: FontWeight.w500)),
        const SizedBox(height: 4),
        Text(
          value, 
          style: TextStyle(
            fontWeight: FontWeight.bold, 
            fontSize: 15, 
            color: color ?? const Color(0xFF2D3142)
          )
        ),
      ],
    );
  }
}
