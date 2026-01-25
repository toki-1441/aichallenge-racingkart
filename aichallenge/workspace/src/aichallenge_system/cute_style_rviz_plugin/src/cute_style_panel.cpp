#include "cute_style_rviz_plugin/cute_style_panel.hpp"

#include <fstream>
#include <sstream>

#include <QApplication>
#include <QCoreApplication>
#include <QFileDialog>

#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction.hpp>

#include <pluginlib/class_list_macros.hpp>

namespace cute_style_rviz_plugin
{

CuteStylePanel::CuteStylePanel(QWidget * parent)
: rviz_common::Panel(parent),
  theme_topic_name_("/cute_style_rviz_plugin/theme_qss"),
  title_label_(new QLabel(this)),
  status_label_(new QLabel(this)),
  theme_path_edit_(new QLineEdit(this)),
  browse_button_(new QPushButton(tr("Browse…"), this)),
  apply_file_button_(new QPushButton(tr("Apply From File"), this)),
  apply_default_button_(new QPushButton(tr("Apply Cute Default"), this)),
  reset_button_(new QPushButton(tr("Reset Theme"), this)),
  subscribe_checkbox_(new QCheckBox(tr("Subscribe theme topic (std_msgs/String QSS)"), this)),
  topic_name_edit_(new QLineEdit(this))
{
  title_label_->setText(tr("Cute Style RViz Plugin"));
  title_label_->setObjectName("CuteTitleLabel");

  status_label_->setText(tr("Status: ready"));
  status_label_->setWordWrap(true);
  status_label_->setObjectName("CuteStatusLabel");

  theme_path_edit_->setPlaceholderText(tr("Theme file path (.qss)"));

  topic_name_edit_->setText(QString::fromStdString(theme_topic_name_));
  topic_name_edit_->setPlaceholderText(tr("Theme topic name"));

  subscribe_checkbox_->setChecked(true);

  auto * root = new QVBoxLayout();
  root->addWidget(title_label_);

  auto * file_row = new QHBoxLayout();
  file_row->addWidget(theme_path_edit_);
  file_row->addWidget(browse_button_);
  root->addLayout(file_row);

  auto * button_row = new QHBoxLayout();
  button_row->addWidget(apply_default_button_);
  button_row->addWidget(apply_file_button_);
  button_row->addWidget(reset_button_);
  root->addLayout(button_row);

  auto * topic_row = new QHBoxLayout();
  topic_row->addWidget(topic_name_edit_);
  root->addLayout(topic_row);
  root->addWidget(subscribe_checkbox_);
  root->addWidget(status_label_);
  root->addStretch(1);
  setLayout(root);

  connect(apply_default_button_, &QPushButton::clicked, this, &CuteStylePanel::applyDefaultTheme);
  connect(apply_file_button_, &QPushButton::clicked, this, &CuteStylePanel::applyThemeFromFile);
  connect(browse_button_, &QPushButton::clicked, this, &CuteStylePanel::browseThemeFile);
  connect(reset_button_, &QPushButton::clicked, this, &CuteStylePanel::resetTheme);
  connect(
    subscribe_checkbox_, &QCheckBox::stateChanged, this, &CuteStylePanel::toggleTopicSubscription);
  connect(topic_name_edit_, &QLineEdit::editingFinished, this, &CuteStylePanel::topicNameEdited);
}

void CuteStylePanel::onInitialize()
{
  auto context = getDisplayContext();
  if (context) {
    auto node_abstraction = context->getRosNodeAbstraction().lock();
    if (node_abstraction) {
      ros_node_ = node_abstraction->get_raw_node();
    }
  }

  ensureRosNode();
  ensureThemeSubscription();

  setStatus(tr("Status: loaded (topic: %1)").arg(QString::fromStdString(theme_topic_name_)));
}

void CuteStylePanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  config.mapSetValue("theme_path", theme_path_edit_->text());
  config.mapSetValue("topic_name", topic_name_edit_->text());
  config.mapSetValue("subscribe", subscribe_checkbox_->isChecked());
}

void CuteStylePanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);

  QString theme_path;
  if (config.mapGetString("theme_path", &theme_path)) {
    theme_path_edit_->setText(theme_path);
  }

  QString topic_name;
  if (config.mapGetString("topic_name", &topic_name) && !topic_name.trimmed().isEmpty()) {
    topic_name_edit_->setText(topic_name);
  }

  bool subscribe = true;
  config.mapGetBool("subscribe", &subscribe);
  subscribe_checkbox_->setChecked(subscribe);

  dropThemeSubscription();
  ensureRosNode();
  if (subscribe_checkbox_->isChecked()) {
    ensureThemeSubscription();
  }
}

void CuteStylePanel::ensureRosNode()
{
  if (!ros_node_) {
    ros_node_ = rclcpp::Node::make_shared("cute_style_rviz_panel");
  }
}

void CuteStylePanel::ensureThemeSubscription()
{
  if (!ros_node_) {
    return;
  }

  const auto topic = topic_name_edit_->text().trimmed().toStdString();
  if (topic.empty()) {
    setStatus(tr("Status: topic name is empty"));
    return;
  }
  theme_topic_name_ = topic;

  if (theme_sub_) {
    return;
  }

  const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).durability_volatile().best_effort();
  theme_sub_ = ros_node_->create_subscription<std_msgs::msg::String>(
    theme_topic_name_, qos,
    [this](const std_msgs::msg::String::SharedPtr msg)
    {
      if (!msg) {
        return;
      }
      const QString qss = QString::fromStdString(msg->data);
      if (qss.trimmed().isEmpty()) {
        setStatus(tr("Status: received empty QSS (ignored)"));
        return;
      }
      applyStyleSheet(qss, tr("topic: %1").arg(QString::fromStdString(theme_topic_name_)));
    });
}

void CuteStylePanel::dropThemeSubscription()
{
  theme_sub_.reset();
}

void CuteStylePanel::toggleTopicSubscription(int state)
{
  if (state == Qt::Checked) {
    dropThemeSubscription();
    ensureThemeSubscription();
    return;
  }
  dropThemeSubscription();
  setStatus(tr("Status: topic subscription disabled"));
}

void CuteStylePanel::topicNameEdited()
{
  dropThemeSubscription();
  if (subscribe_checkbox_->isChecked()) {
    ensureThemeSubscription();
    return;
  }
  setStatus(tr("Status: topic updated (subscription disabled)"));
}

void CuteStylePanel::applyDefaultTheme()
{
  applyStyleSheet(defaultThemeQss(), tr("built-in default"));
}

void CuteStylePanel::applyThemeFromFile()
{
  const QString path = theme_path_edit_->text().trimmed();
  if (path.isEmpty()) {
    setStatus(tr("Status: theme path is empty"));
    return;
  }

  QString error;
  const QString qss = loadTextFile(path, error);
  if (!error.isEmpty()) {
    setStatus(tr("Status: failed to read theme file: %1").arg(error));
    return;
  }

  applyStyleSheet(qss, tr("file: %1").arg(path));
}

void CuteStylePanel::browseThemeFile()
{
  const QString path = QFileDialog::getOpenFileName(
    this, tr("Select a .qss theme file"), QString(), tr("Qt Stylesheet (*.qss);;All Files (*)"));
  if (path.isEmpty()) {
    return;
  }
  theme_path_edit_->setText(path);
}

void CuteStylePanel::resetTheme()
{
  if (auto * app = qobject_cast<QApplication *>(QCoreApplication::instance())) {
    app->setStyleSheet(QString());
  }
  setStatus(tr("Status: theme reset"));
}

void CuteStylePanel::setStatus(const QString & text)
{
  status_label_->setText(text);
}

void CuteStylePanel::applyStyleSheet(const QString & qss, const QString & source)
{
  if (auto * app = qobject_cast<QApplication *>(QCoreApplication::instance())) {
    app->setStyleSheet(qss);
  }
  setStatus(tr("Status: applied theme (%1)").arg(source));
}

QString CuteStylePanel::loadTextFile(const QString & path, QString & error) const
{
  error.clear();
  std::ifstream file(path.toStdString(), std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    error = tr("cannot open: %1").arg(path);
    return {};
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return QString::fromStdString(ss.str());
}

QString CuteStylePanel::defaultThemeQss() const
{
  // Keep this compact: the “real” theme file lives in share/<pkg>/themes.
  // This built-in theme is a safe fallback for quick testing.
  return QStringLiteral(R"QSS(
/* Cute Style RViz Plugin - built-in fallback theme */
QWidget {
  background-color: #1b1026;
  color: #ffe7ff;
  font-size: 12px;
}

QMainWindow, QDockWidget {
  background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2a1140, stop:1 #1b1026);
}

QLabel#CuteTitleLabel {
  font-size: 16px;
  font-weight: 700;
  padding: 8px 10px;
  border-radius: 10px;
  background-color: rgba(255, 105, 180, 0.18);
  border: 1px solid rgba(255, 105, 180, 0.45);
}

QLabel#CuteStatusLabel {
  padding: 8px;
  border-radius: 10px;
  background-color: rgba(186, 104, 200, 0.14);
  border: 1px solid rgba(186, 104, 200, 0.35);
}

QLineEdit {
  padding: 7px 10px;
  border-radius: 10px;
  border: 1px solid rgba(255, 105, 180, 0.35);
  background-color: rgba(255, 255, 255, 0.06);
}

QPushButton {
  padding: 7px 12px;
  border-radius: 12px;
  border: 1px solid rgba(255, 105, 180, 0.45);
  background-color: rgba(255, 105, 180, 0.16);
}
QPushButton:hover {
  background-color: rgba(186, 104, 200, 0.22);
  border: 1px solid rgba(186, 104, 200, 0.55);
}
QPushButton:pressed {
  background-color: rgba(255, 105, 180, 0.26);
}

QCheckBox {
  spacing: 8px;
  padding: 6px;
}
QCheckBox::indicator {
  width: 16px;
  height: 16px;
  border-radius: 5px;
  border: 1px solid rgba(255, 105, 180, 0.55);
  background: rgba(255,255,255,0.06);
}
QCheckBox::indicator:checked {
  background: rgba(255,105,180,0.65);
}
  )QSS");
}

}  // namespace cute_style_rviz_plugin

PLUGINLIB_EXPORT_CLASS(
  cute_style_rviz_plugin::CuteStylePanel,
  rviz_common::Panel)
