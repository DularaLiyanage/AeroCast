import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../utils/forecast_utils.dart';

class XaiSection extends StatelessWidget {
  final Map<String, dynamic> rawXaiData;

  const XaiSection({super.key, required this.rawXaiData});

  @override
  Widget build(BuildContext context) {
    final drivers = ForecastUtils.processXaiDrivers(rawXaiData, topN: 5);
    if (drivers.isEmpty) return const SizedBox.shrink();

    final breakdown = ForecastUtils.getXaiCategoryBreakdown(rawXaiData);
    final dominantCategory = ForecastUtils.getDominantXaiCategory(rawXaiData);
    final tip = ForecastUtils.getXaiContextualTip(dominantCategory);
    final catColor = ForecastUtils.xaiCategoryColors[dominantCategory] ?? Colors.blue;
    final catIcon = ForecastUtils.xaiCategoryIcons[dominantCategory] ?? Icons.info_outline;
    final maxPct = drivers.first.value;

    // Build donut sections from category breakdown (skip < 1%)
    final donutSections = breakdown.entries
        .where((e) => e.value >= 1.0)
        .map((e) {
          final color = ForecastUtils.xaiCategoryColors[e.key] ?? Colors.grey;
          return PieChartSectionData(
            value: e.value,
            color: color,
            radius: 22,
            showTitle: false,
          );
        })
        .toList();

    // Legend entries sorted by value descending
    final legendEntries = breakdown.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 20),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 20,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Header ──
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.blue[50],
                  shape: BoxShape.circle,
                ),
                child: Icon(Icons.analytics_outlined, color: Colors.blue[700], size: 20),
              ),
              const SizedBox(width: 12),
              const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Why this forecast?",
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                  Text("Top factors influencing this prediction",
                      style: TextStyle(fontSize: 12, color: Colors.grey)),
                ],
              ),
            ],
          ),

          const SizedBox(height: 20),

          // ── Donut chart + category legend ──
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // Donut
              Stack(
                alignment: Alignment.center,
                children: [
                  SizedBox(
                    height: 110,
                    width: 110,
                    child: PieChart(
                      PieChartData(
                        sections: donutSections,
                        centerSpaceRadius: 35,
                        sectionsSpace: 2,
                        startDegreeOffset: -90,
                      ),
                    ),
                  ),
                  Icon(catIcon, color: catColor, size: 22),
                ],
              ),

              const SizedBox(width: 20),

              // Legend
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: legendEntries.map((e) {
                    final color = ForecastUtils.xaiCategoryColors[e.key] ?? Colors.grey;
                    final icon = ForecastUtils.xaiCategoryIcons[e.key] ?? Icons.circle;
                    final isDominant = e.key == dominantCategory;
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 10),
                      child: Row(
                        children: [
                          Container(
                            width: 10,
                            height: 10,
                            decoration: BoxDecoration(
                              color: color,
                              borderRadius: BorderRadius.circular(2),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Icon(icon, size: 13, color: color),
                          const SizedBox(width: 5),
                          Expanded(
                            child: Text(
                              e.key,
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: isDominant ? FontWeight.bold : FontWeight.normal,
                                color: isDominant ? Colors.black87 : Colors.black54,
                              ),
                            ),
                          ),
                          Text(
                            "${e.value.toStringAsFixed(1)}%",
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.bold,
                              color: color,
                            ),
                          ),
                        ],
                      ),
                    );
                  }).toList(),
                ),
              ),
            ],
          ),

          const SizedBox(height: 16),

          // ── Contextual tip ──
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            decoration: BoxDecoration(
              color: catColor.withValues(alpha: 0.07),
              borderRadius: BorderRadius.circular(12),
              border: Border(left: BorderSide(color: catColor, width: 3)),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(catIcon, size: 15, color: catColor),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    tip,
                    style: const TextStyle(
                        fontSize: 13, color: Colors.black87, height: 1.45),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 20),

          // ── Divider before driver bars ──
          Row(
            children: [
              const Expanded(child: Divider()),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 10),
                child: Text("Top Factors",
                    style: TextStyle(fontSize: 12, color: Colors.grey[500])),
              ),
              const Expanded(child: Divider()),
            ],
          ),

          const SizedBox(height: 16),

          // ── Driver bars ──
          ...drivers.map((entry) => _DriverRow(
                name: entry.key,
                pct: entry.value,
                barFraction: maxPct > 0
                    ? (entry.value / maxPct).clamp(0.0, 1.0)
                    : 0.0,
              )),
        ],
      ),
    );
  }
}

class _DriverRow extends StatelessWidget {
  final String name;
  final double pct;
  final double barFraction;

  const _DriverRow({
    required this.name,
    required this.pct,
    required this.barFraction,
  });

  @override
  Widget build(BuildContext context) {
    final category =
        ForecastUtils.xaiDriverCategories[name] ?? 'Weather';
    final color =
        ForecastUtils.xaiCategoryColors[category] ?? Colors.blue;
    final icon =
        ForecastUtils.xaiCategoryIcons[category] ?? Icons.info_outline;

    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, size: 15, color: color),
              const SizedBox(width: 6),
              Expanded(
                child: Text(name,
                    style: const TextStyle(
                        fontWeight: FontWeight.w600, fontSize: 14)),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(category,
                    style: TextStyle(
                        fontSize: 10,
                        color: color,
                        fontWeight: FontWeight.w700)),
              ),
              const SizedBox(width: 8),
              SizedBox(
                width: 42,
                child: Text("${pct.toStringAsFixed(1)}%",
                    textAlign: TextAlign.end,
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 13,
                        color: color)),
              ),
            ],
          ),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: barFraction,
              backgroundColor: Colors.grey[100],
              color: color,
              minHeight: 6,
            ),
          ),
        ],
      ),
    );
  }
}
