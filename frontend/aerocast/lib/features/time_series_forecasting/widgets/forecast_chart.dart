import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';

class ForecastChart extends StatelessWidget {
  final List<dynamic> values;
  final String pollutant;
  final int selectedHourIndex;           // <--- Received from Parent
  final Function(int) onHourChanged;     // <--- Callback to Parent

  const ForecastChart({
    super.key, 
    required this.values, 
    required this.pollutant,
    required this.selectedHourIndex,     // <--- Required
    required this.onHourChanged,         // <--- Required
  });

  DateTime _getTomorrowStart() {
    DateTime now = DateTime.now();
    return DateTime(now.year, now.month, now.day).add(const Duration(days: 1));
  }

  @override
  Widget build(BuildContext context) {
    List<FlSpot> spots = [];
    for (int i = 0; i < values.length; i++) {
      spots.add(FlSpot(i.toDouble(), values[i].toDouble()));
    }

    LineChartBarData lineBarData = LineChartBarData(
      spots: spots,
      isCurved: true,
      preventCurveOverShooting: true,
      color: Colors.black,
      barWidth: 2.5,
      dotData: FlDotData(show: false),
      belowBarData: BarAreaData(
        show: true,
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Colors.black.withOpacity(0.12), Colors.transparent],
        ),
      ),
    );

    return Container(
      height: 320,
      margin: const EdgeInsets.symmetric(horizontal: 20),
      padding: const EdgeInsets.fromLTRB(10, 20, 20, 10),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(30),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Padding(
                padding: EdgeInsets.only(left: 10, bottom: 10),
                child: Text("24h Trend", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(color: Colors.black, borderRadius: BorderRadius.circular(12)),
                child: Text(
                   "${values[selectedHourIndex].toStringAsFixed(1)} µg/m³",
                   style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 12),
                ),
              )
            ],
          ),
          
          Expanded(
            child: LineChart(
              LineChartData(
                minY: 0,
                gridData: const FlGridData(show: false),
                titlesData: FlTitlesData(
                  leftTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      interval: 6,
                      getTitlesWidget: (val, meta) {
                        int hour = val.toInt();
                        if (hour >= 24) return const SizedBox();
                        DateTime time = _getTomorrowStart().add(Duration(hours: hour));
                        return Padding(
                          padding: const EdgeInsets.only(top: 8.0),
                          child: Text(DateFormat('ha').format(time),
                              style: const TextStyle(color: Colors.grey, fontSize: 12)),
                        );
                      },
                    ),
                  ),
                ),
                borderData: FlBorderData(show: false),
                
                // HIGHLIGHT SELECTED SPOT
                showingTooltipIndicators: [
                  ShowingTooltipIndicators([
                    LineBarSpot(lineBarData, 0, lineBarData.spots[selectedHourIndex]),
                  ]),
                ],
                
                lineTouchData: LineTouchData(
                  enabled: false, // Touch disabled, Slider controls it
                  getTouchedSpotIndicator: (barData, spotIndexes) {
                    return spotIndexes.map((index) {
                      return TouchedSpotIndicatorData(
                        const FlLine(color: Colors.black, strokeWidth: 2, dashArray: [4, 4]),
                        FlDotData(
                          show: true,
                          getDotPainter: (spot, percent, bar, index) =>
                              FlDotCirclePainter(radius: 6, color: Colors.black, strokeWidth: 2, strokeColor: Colors.white),
                        ),
                      );
                    }).toList();
                  },
                  touchTooltipData: LineTouchTooltipData(
                    tooltipBgColor: Colors.black,
                    getTooltipItems: (List<LineBarSpot> touchedBarSpots) {
                      return touchedBarSpots.map((barSpot) {
                        return LineTooltipItem(
                          DateFormat('ha').format(_getTomorrowStart().add(Duration(hours: barSpot.x.toInt()))),
                          const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                        );
                      }).toList();
                    },
                  ),
                ),
                lineBarsData: [lineBarData],
              ),
            ),
          ),

          const SizedBox(height: 10),

          // SLIDER
          Column(
            children: [
              const Text("Slide to see specific hour", style: TextStyle(color: Colors.grey, fontSize: 10)),
              SizedBox(
                height: 30,
                child: SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    trackHeight: 2,
                    thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 6),
                    overlayShape: const RoundSliderOverlayShape(overlayRadius: 14),
                  ),
                  child: Slider(
                    value: selectedHourIndex.toDouble(),
                    min: 0,
                    max: (values.length - 1).toDouble(),
                    divisions: values.length > 1 ? values.length - 1 : 1,
                    activeColor: Colors.black,
                    inactiveColor: Colors.grey[200],
                    onChanged: (val) {
                      onHourChanged(val.toInt()); // <--- Notify Parent
                    },
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}